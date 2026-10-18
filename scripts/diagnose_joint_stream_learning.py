# -*- coding: utf-8 -*-
"""Contract-bound ONE final64 reference trajectory per fixed witness.

Not executed by authorship. Requires pre-existing frozen contract.json. No resume,
rerun, overwrite, optimizer, policy update, alternative-policy rollout or training.
All new outputs live under --output-root. Original science artifacts are read-only.
"""
import sys
sys.dont_write_bytecode=True
import argparse
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import gzip,hashlib,importlib.util,json,os,random,traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import joint_stream_same_state_scoring as score

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(r):return json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
def write_json(p,x,*,exclusive=False):
 with Path(p).open('x' if exclusive else 'w') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def read_rows(p):
 with Path(p).open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def key(w):return (w.get('method',w.get('agent')),int(w['seed']),w['case_id'])
def forbidden(*args,**kwargs):raise RuntimeError('Stage2 forbids every policy update and optimizer step')

@contextmanager
def no_training(models):
 with ExitStack() as stack:
  for m in models.values():stack.enter_context(score.patched_method(m,'update',forbidden))
  # All live stream variants use Adam. Guard the base optimizer too.
  orig_adam=torch.optim.Adam.step;orig_base=torch.optim.Optimizer.step
  torch.optim.Adam.step=forbidden;torch.optim.Optimizer.step=forbidden
  try:yield
  finally:torch.optim.Adam.step=orig_adam;torch.optim.Optimizer.step=orig_base


def verify_contract(root,out):
 p=out/'contract.json';contract=json.loads(p.read_text())
 required=['witnesses','source_run','source_commit','max_reference_frames','max_reference_sessions','max_frames_per_session','max_extra_frames','script_sha256']
 if any(k not in contract for k in required):raise ValueError('Missing frozen machine contract keys')
 if contract['max_reference_frames']!=7200 or contract['max_reference_sessions']!=15 or contract['max_frames_per_session']!=480 or contract['max_extra_frames']!=8:raise ValueError('Unapproved stage2 cap')
 witnesses=contract['witnesses'];ident=[key(w) for w in witnesses]
 if len(witnesses)!=15 or len(set(ident))!=15:raise ValueError('Expected exactly15unique frozen witnesses')
 if not all(m in {'sa_ghmappo','cache_offload_drl','dqn','mappo','dag_offload_drl'} for m,s,c in ident):raise ValueError('Unexpected witness method')
 for name,expected in contract['script_sha256'].items():
  path=Path(name) if Path(name).is_absolute() else root/name
  if sha(path)!=expected:raise ValueError('Frozen stage2 script hash mismatch: '+name)
 if contract.get('score_snapshots',[0,32,64])!=[0,32,64] or contract.get('score_variants',['0','32','64','prior_only_64'])!=['0','32','64','prior_only_64']:raise ValueError('Unapproved scoring variants')
 if contract.get('stage1_task_points'):
  pointspath=Path(contract['stage1_task_points'])
  if sha(pointspath)!=contract['stage1_task_points_sha256']:raise ValueError('Stage1 fixed witness plan changed')
  if json.loads(pointspath.read_text())['fixed_witnesses']['cases']!=witnesses:raise ValueError('Witness list differs from preserved stage1 plan')
 if contract.get('contract_document') and sha(root/contract['contract_document'])!=contract['contract_document_sha256']:raise ValueError('Frozen stage2 document changed')
 source=Path(contract['source_run']);protocol=json.loads((source/'protocol/manifest.json').read_text())
 if protocol['source_commit']!=contract['source_commit'] or protocol['action_contract']!='joint_stream_control_v2':raise ValueError('Frozen algorithm identity mismatch')
 checks=[]
 for name,expected in protocol['source_fingerprints'].items():
  path=root/name;actual=sha(path);checks.append(dict(path=str(path),sha256=actual))
  if actual!=expected:raise ValueError('Frozen science source changed: '+name)
 assets=[]
 for name,expected in protocol['assets'].items():
  path=source/'protocol'/name;actual=sha(path)
  if actual!=expected:raise ValueError('Frozen protocol asset changed: '+name)
  assets.append(dict(path=str(path),sha256=actual))
 original_rows={};manifests={};checkpoint_checks=[]
 for method in protocol['methods']:
  d=source/method;man=json.loads((d/'manifest.json').read_text());manifests[method]=man
  if man['status']!='complete' or man['source_commit']!=contract['source_commit'] or man['git_dirty']:raise ValueError('Source method provenance invalid')
  if sha(source/'protocol/manifest.json')!=man['protocol_sha256']:raise ValueError('Source protocol hash mismatch')
  if sha(d/'dev.jsonl')!=man['output_sha256']['dev.jsonl']:raise ValueError('Source dev changed: '+method)
  for cp in man['checkpoints']:
   if sha(d/cp['path'])!=cp['sha256']:raise ValueError('Source checkpoint changed')
   checkpoint_checks.append(dict(method=method,seed=cp['seed'],episode=cp['episode'],sha256=cp['sha256']))
  for row in read_rows(d/'dev.jsonl'):
   if row['episode']==64:original_rows[method,row['seed'],row['case_id']]=row
 for ident in map(key,witnesses):
  if ident not in original_rows:raise ValueError('Witness not in immutable final64 evidence')
 # Check declared protocol hash if machine contract includes it.
 if contract.get('source_protocol_sha256',sha(source/'protocol/manifest.json'))!=sha(source/'protocol/manifest.json'):raise ValueError('Machine protocol hash mismatch')
 return contract,protocol,manifests,original_rows,dict(checked_at=utc(),contract_sha256=sha(p),science_source_checks=len(checks),source_files=checks,protocol_assets=assets,checkpoint_count=len(checkpoint_checks),checkpoint_checks=checkpoint_checks,source_manifests={m:sha(source/m/'manifest.json') for m in protocol['methods']},source_protocol_sha256=sha(source/'protocol/manifest.json'))


def source_trace_digests(source,manifests,witnesses):
 wanted={key(w) for w in witnesses};out={k:{} for k in wanted}
 for method in sorted({k[0] for k in wanted}):
  wanted_for={k for k in wanted if k[0]==method}
  for filename in ['events.jsonl','receipts.jsonl']:
   per={k:hashlib.sha256() for k in wanted_for};count={k:0 for k in wanted_for};whole=hashlib.sha256()
   with (source/method/filename).open('rb') as f:
    for line in f:
     whole.update(line)
     # Source identity appears before event body. Avoid parsing non-witness rows.
     if not any((f'"seed": {seed},'.encode() in line and f'"case_id": "{case}"'.encode() in line) for _,seed,case in wanted_for):continue
     row=json.loads(line);k=(row['agent'],row['seed'],row['case_id'])
     if k not in per:continue
     if row['episode']!=64:raise ValueError('Unexpected nonfinal source raw trace')
     per[k].update(canonical(row));count[k]+=1
   if whole.hexdigest()!=manifests[method]['output_sha256'][filename]:raise ValueError('Source raw file hash mismatch: '+method+'/'+filename)
   for k in wanted_for:out[k][filename]=dict(canonical_sha256=per[k].hexdigest(),rows=count[k])
 return out


def gzip_digest(path):
 h=hashlib.sha256();n=0
 with gzip.open(path,'rt') as f:
  for line in f:h.update(canonical(json.loads(line)));n+=1
 return dict(canonical_sha256=h.hexdigest(),rows=n)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output_root','--output-root',dest='output_root',type=Path,required=True);p.add_argument('--repo_root','--repo-root',dest='repo_root',type=Path,default=ROOT);a=p.parse_args()
 out=a.output_root.resolve();root=a.repo_root.resolve();out.mkdir(parents=True,exist_ok=True)
 # A root-created contract is required. The exclusive claim prevents any rerun.
 if not (out/'contract.json').is_file():raise ValueError('Frozen contract.json required before execution')
 write_json(out/'run_started.json',dict(started_at=utc(),pid=os.getpid(),status='verification',training_sessions=0,optimizer_steps=0),exclusive=True)
 current=None;used_frames=0;completed=[];captures=[];env=None
 try:
  contract,protocol,manifests,original_rows,verification=verify_contract(root,out);source=Path(contract['source_run'])
  if str(root) not in sys.path:sys.path.insert(0,str(root))
  from scripts.train_joint_stream_control import make_env,stream_episode
  torch.set_num_threads(1)
  data=torch.load(source/'protocol/protocol.pt',map_location='cpu',weights_only=False)
  cases={c['case_id']:c for c in data['cases']['dev']}
  for w in contract['witnesses']:
   if len(data['sessions'][cases[w['case_id']]['session']]['frames'])>contract['max_frames_per_session']:raise ValueError('Case source exceeds reference frame cap')
  source_digests=source_trace_digests(source,manifests,contract['witnesses'])
  verification['source_raw_trace_hashes']={str(k):v for k,v in source_digests.items()};write_json(out/'verification.json',verification,exclusive=True)
  loaded={}
  for index,w in enumerate(contract['witnesses']):
   method,seed,case_id=key(w);current=dict(index=index,method=method,seed=seed,case_id=case_id);env=None
   group=out/f'{index:02d}_{method}_seed{seed}_{case_id}';group.mkdir(exist_ok=False)
   if used_frames+480>contract['max_reference_frames']:raise ValueError('Remaining frame authorization insufficient')
   if (method,seed) not in loaded:loaded[method,seed]=score.load_exact_models(root,source,method,seed)
   models,provenance=loaded[method,seed];identity=dict(agent=method,rule=f'{method}_seed{seed}',seed=seed,episode=64,case_id=case_id)
   weights={k:score.module_digest(v) for k,v in models.items()};random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   with no_training(models):
    env=make_env(cases[case_id],data,protocol)
    with gzip.open(group/'events.jsonl.gz','xt',encoding='utf8') as eventfile,gzip.open(group/'receipts.jsonl.gz','xt',encoding='utf8') as receiptfile:
     with score.CaptureTap(models['64'],dict(method=method,seed=seed,case_id=case_id),max_extra_frames=8) as tap:
      row,rolls=stream_episode(env,models['64'],deterministic=True,event_file=eventfile,receipt_file=receiptfile,identity=identity)
    used_frames+=row['steps']
    if row['steps']>480 or used_frames>contract['max_reference_frames']:raise ValueError('Reference physical frame cap exceeded')
    actual=dict(identity,**row);expected=original_rows[method,seed,case_id]
    if actual!=expected:
     diff=[k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)]
     write_json(group/'row_parity_failure.json',dict(different_fields=diff,actual=actual,expected=expected),exclusive=True)
     raise ValueError('Reference fullrow differs from immutable64: '+str(diff))
    trace_checks={}
    for name in ['events.jsonl','receipts.jsonl']:
     got=gzip_digest(group/(name+'.gz'));trace_checks[name]=got
     if got!=source_digests[method,seed,case_id][name]:raise ValueError('Reference event/receipt fullcontent parity failed: '+name)
    if len(tap.selected_frames)>9 or len(tap.snapshots)>144:raise ValueError('Perwitness capture cap exceeded')
    if sum(z['snapshot_count'] for z in completed)+len(tap.snapshots)>contract.get('max_saved_calls',2160):raise ValueError('Total capture cap exceeded')
    if sum(z['second_pass_rows'] for z in completed)+sum(s['phase']=='second_cache_event_pass' for s in tap.snapshots)>contract.get('max_saved_second_pass_rows',1080):raise ValueError('Secondpass cap exceeded')
    tap.bind_episode_labels(rolls,fleet=8);tap.save(group/'snapshots.pt')
    selected_rolls=[r for r in rolls if int(r['frame_index']) in tap.selected_frames]
    selected_decisions=sum(len(r['actions']) for r in selected_rolls)
    if selected_decisions!=sum(s['phase']=='second_cache_event_pass' for s in tap.snapshots):raise ValueError('Selectedroll/capturecountmismatch')
    torch.save(dict(schema='reference64_selected_rollouts_v1',witness=current,rollouts=selected_rolls,initial_request_count=8,episode_decision_rows=row['decision_rows'],selected_decision_rows=selected_decisions,selected_frames=tap.selected_frames,MC_contract='labelscomputedbyunchangedcompleteepisode;undiscountedphysicalteamtail/10;teamcredit/8;deterministicpolicymassisnottruebehaviorprob'),group/'rollouts.pt')
    with gzip.open(group/'scoring.jsonl.gz','xt',encoding='utf8') as f:
     for snap in tap.snapshots:f.write(json.dumps(score.score_snapshot(snap,models),ensure_ascii=False)+'\n')
   if weights!={k:score.module_digest(v) for k,v in models.items()}:raise ValueError('Reference/scoring changed model weights')
   write_json(group/'reference_row.json',actual,exclusive=True)
   result=dict(status='pass',completed_at=utc(),witness=current,reference_frames=row['steps'],selected_frames=tap.selected_frames,
    snapshot_count=len(tap.snapshots),second_pass_rows=sum(s['phase']=='second_cache_event_pass' for s in tap.snapshots),
    episode_decision_rows=row['decision_rows'],selected_decision_rows=selected_decisions,fleet=8,reference_summary_exact=True,reference_traces_exact=trace_checks,weights_unchanged=True,provenance=provenance,
    output_sha256={f:sha(group/f) for f in ['events.jsonl.gz','receipts.jsonl.gz','snapshots.pt','rollouts.pt','scoring.jsonl.gz','reference_row.json']},training_sessions=0,optimizer_steps=0)
   write_json(group/'result.json',result,exclusive=True);completed.append(result)
   captures.append(dict(path=str(group/'snapshots.pt'),sha256=result['output_sha256']['snapshots.pt'],witness=current))
   write_json(out/'progress.json',dict(status='running',updated_at=utc(),completed_sessions=len(completed),reference_frames=used_frames,current=current,captures=captures,training_sessions=0,optimizer_steps=0))
   print(json.dumps(dict(phase='witness_complete',witness=current,completed=len(completed),reference_frames=used_frames)),flush=True)
  result=dict(status='complete',completed_at=utc(),contract_sha256=sha(out/'contract.json'),reference_sessions=len(completed),reference_frames=used_frames,
   max_reference_frames=7200,captures=captures,results=completed,training_sessions=0,optimizer_steps=0,new_policy_rollouts='one frozenfinal64reference per fixedwitness only',
   semantic_boundary='same-state conditional score comparison under reference64teacherforcedprefix,notalternativepolicyperformance')
  write_json(out/'result.json',result,exclusive=True);write_json(out/'progress.json',dict(status='complete',completed_at=utc(),completed_sessions=len(completed),reference_frames=used_frames,captures=captures))
 except BaseException as exc:
  failure=dict(status='failed',failed_at=utc(),phase=current or 'verification',error=repr(exc),traceback=traceback.format_exc(),completed_sessions=len(completed),known_completed_reference_frames=used_frames,
    current_env_frame=getattr(env,'frame_index',None),training_sessions=0,optimizer_steps=0,policy='preserve scene; no rerun/overwrite')
  if not (out/'failure.json').exists():write_json(out/'failure.json',failure,exclusive=True)
  write_json(out/'progress.json',failure);raise


if __name__=='__main__':main()
