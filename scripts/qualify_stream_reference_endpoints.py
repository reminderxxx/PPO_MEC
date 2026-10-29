"""Qualify only frozen main final60 native endpoints, without simulation.

The legacy reduced controls lack node_id/workflow_completed. Delivered-node
statuses are counted per request and bound to the frozen workflow node totals.
This is a targeted reference-endpoint check, not a rerun of the reward/MC audit.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

CAMPAIGN_SHA='abd965ec2617cae544ed38f9cf607f9c9c87383e324dda8211783f46824357e1'
AUDIT_SHA='d795a848637b7828214cd41582387075e4ad50a7269682a4d756674165f1a948'
SOURCE_COMMIT='e00d5faf9c3f73a689174237245ed6f85394b2f2'
SEEDS=[7,17,27,37,47]
CASES=['train_000','train_014','train_019','train_007','train_024','train_038',
       'train_043','train_031','train_048','train_062','train_067','train_055']
EVENT_KEYS=('frame_index','vehicle_id','request_id','workflow_id','status','control',
    'control_planning','cache_request','state_transfer_link_frames','execution_rsu_id',
    'service_target_changed','association_handoff')
DELIVERED_STATUSES={'node_completed','vehicle_fallback_completed'}
KNOWN_STATUSES={'waiting','parents_not_ready','state_unavailable_no_coverage',
    'state_restore','state_link_queue','vehicle_fallback_completed','vehicle_compute',
    'adapter_not_ready','input_available','input_link_queue','input_ready','input_transfer',
    'compute_queue','output_ready','rsu_compute','output_out_of_coverage',
    'output_link_queue','output_transfer','node_completed'}
ENV_SOURCES=('src/envs/core/multi_request_stream_env.py',
    'src/envs/core/multi_request_target_control_env.py',
    'src/envs/core/multi_request_joint_control_env.py')
CAPS=dict(maximum_input_bytes=2*1024**3,maximum_decoded_bytes=8*1024**3,
          maximum_wall_seconds=1800,maximum_output_bytes=32*1024**2)


def packed(value):
    return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n'


def close(actual,expected,label):
    if (type(actual) not in (int,float) or type(expected) not in (int,float)
            or not math.isfinite(actual) or not math.isfinite(expected)
            or abs(actual-expected)>1e-6):
        raise ValueError(label+' differs or unavailable')


def nonnegative(value,label):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError(label+' negative, nonfinite or unavailable')
    return value


class ReadBudget:
    """All source bytes (including metadata) count against the2GiB read ceiling."""
    def __init__(self,output,caps=None):
        self.output=Path(output);self.caps=dict(CAPS if caps is None else caps)
        self.started=time.monotonic();self.input_bytes=0;self.compressed_bytes=0
        self.decoded_bytes=0;self.files=[]

    def check(self):
        if time.monotonic()-self.started>=self.caps['maximum_wall_seconds']:
            raise RuntimeError('reference endpoint wall budget exhausted')

    def charge(self,n,*,compressed=False):
        self.check()
        if self.input_bytes+n>self.caps['maximum_input_bytes']:
            raise RuntimeError('reference input read budget exhausted')
        self.input_bytes+=n
        if compressed:self.compressed_bytes+=n

    def read(self,path,*,keep=True):
        """Hash one non-gzip source; bytes are optionally retained for parsing."""
        path=Path(path);digest=hashlib.sha256();blocks=[]
        with path.open('rb') as stream:
            while True:
                self.check()
                remaining=self.caps['maximum_input_bytes']-self.input_bytes
                if remaining==0 and stream.tell()<path.stat().st_size:
                    raise RuntimeError('reference input read budget exhausted')
                block=stream.read(min(1024*1024,remaining)) if remaining else b''
                if not block:break
                self.charge(len(block));digest.update(block)
                if keep:blocks.append(block)
        record=dict(path=str(path.resolve()),sha256=digest.hexdigest(),bytes=path.stat().st_size,
                    operation='read_only_hash_and_parse' if keep else 'read_only_hash',compressed=False)
        self.files.append(record)
        return b''.join(blocks) if keep else None,record

    def gzip_rows(self,path,expected_sha):
        """One pass computes compressed SHA and decoded accounting together."""
        path=Path(path);budget=self;digest=hashlib.sha256();size=path.stat().st_size
        class Reader:
            def __init__(self,stream):self.stream=stream
            def read(self,n=-1):
                budget.check();remaining=budget.caps['maximum_input_bytes']-budget.input_bytes
                if remaining==0 and self.stream.tell()<size:
                    raise RuntimeError('reference compressed read budget exhausted')
                if n<0:n=remaining
                block=self.stream.read(min(n,remaining))
                budget.charge(len(block),compressed=True);digest.update(block)
                return block
            def seekable(self):return False
        start=self.compressed_bytes;decoded_start=self.decoded_bytes
        with path.open('rb') as raw:
            wrapped=Reader(raw)
            with gzip.GzipFile(fileobj=wrapped,mode='rb') as stream:
                while True:
                    self.check()
                    # Reserve64KiB for the decoder's internal buffered output.
                    remaining=self.caps['maximum_decoded_bytes']-self.decoded_bytes-65536
                    if remaining<=0:raise RuntimeError('reference decoded read budget exhausted')
                    limit=min(16*1024**2,remaining)
                    line=stream.readline(limit)
                    if not line:break
                    self.decoded_bytes+=len(line)
                    if len(line)==limit and not line.endswith(b'\n'):
                        raise RuntimeError('decoded line or remaining read cap exhausted')
                    yield json.loads(line)
            if raw.tell()!=size:
                raise ValueError('compressed source not completely consumed')
        if digest.hexdigest()!=expected_sha:raise ValueError('compressed raw SHA differs')
        self.files.append(dict(path=str(path.resolve()),sha256=digest.hexdigest(),bytes=size,
            operation='single_pass_compressed_hash_and_targeted_final96_decode',compressed=True,
            compressed_bytes_read=self.compressed_bytes-start,decoded_bytes_read=self.decoded_bytes-decoded_start))

    def save(self,path,value):
        self.check();text=packed(value).encode()
        owned=[self.output,Path(str(self.output)+'.claim.json'),Path(str(self.output)+'.failure.json')]
        used=sum(p.stat().st_size for p in owned if p.is_file())
        if used+len(text)+65536>self.caps['maximum_output_bytes']:
            raise RuntimeError('reference endpoint output budget exhausted')
        with Path(path).open('xb') as stream:stream.write(text)


def read_json(path,budget,expected_sha=None):
    raw,record=budget.read(path)
    if expected_sha is not None and record['sha256']!=expected_sha:
        raise ValueError('frozen input SHA differs: '+str(path))
    return json.loads(raw),record


def protocol_cases(data):
    rows=[c for c in data['cases']['train'] if c['case_id'] in CASES]
    if len(rows)!=12 or {c['case_id'] for c in rows}!=set(CASES):
        raise ValueError('original consumed-train case matrix differs')
    result={}
    for case in rows:
        session=data['sessions'][case['session']]
        if session['split']!='train' or not 0<len(session['frames'])<=480:
            raise ValueError('unexpected reference input split/horizon')
        requests={}
        for arrival in case['arrivals']:
            if (arrival.request_id in requests or not arrival.workflow.nodes
                    or type(arrival.frame) is not int or not 0<=arrival.frame<len(session['frames'])):
                raise ValueError('request identity/node count/arrival invalid')
            requests[arrival.request_id]=dict(vehicle_id=arrival.vehicle_id,
                workflow_id=arrival.workflow.workflow_id,arrival_frame=arrival.frame,
                total_nodes=len(arrival.workflow.nodes))
        result[case['case_id']]=dict(horizon=len(session['frames']),requests=requests)
    return result


def qualify_cell(home,cell,campaign,campaign_sha,budget,cases):
    """One old main cell: final12 endpoint check, no old MC or forward replay."""
    seed=cell['seed'];home=Path(home)
    complete,terminal_ref=read_json(home/'complete.json',budget)
    manifest,manifest_ref=read_json(home/'manifest.json',budget)
    if (complete!=manifest or complete['status']!='complete'
            or complete['method']!='main' or complete['seed']!=seed
            or complete['campaign_sha256']!=campaign_sha or complete['source_commit']!=SOURCE_COMMIT
            or (home/'failure.json').exists()):
        raise ValueError('reference main cell terminal/manifest differs')
    required={'protocol.json','training.jsonl','evaluation.jsonl','updates.jsonl',
              'checkpoints.jsonl','receipts.jsonl.gz','controls.jsonl.gz'}
    if set(complete['output_sha256'])!=required:
        raise ValueError('reference raw file inventory differs')
    parsed={};raw_refs=[]
    for name in sorted(required-{'receipts.jsonl.gz','controls.jsonl.gz'}):
        raw,record=budget.read(home/name,keep=name in {'protocol.json','evaluation.jsonl','checkpoints.jsonl'})
        if record['sha256']!=complete['output_sha256'][name]:raise ValueError('reference raw SHA differs')
        raw_refs.append(record)
        if raw is not None:
            parsed[name]=json.loads(raw) if name=='protocol.json' else [json.loads(x) for x in raw.splitlines()]
    if (parsed['protocol.json']['plan']!=campaign
            or parsed['protocol.json']['campaign_sha256']!=campaign_sha):
        raise ValueError('reference embedded scientific plan differs')
    original=[r for r in cell['evaluation'] if r['episode']==96]
    physical=[r for r in parsed['evaluation.jsonl'] if r['episode']==96]
    if len(original)!=12 or len(physical)!=12 or {r['case_id'] for r in physical}!=set(CASES):
        raise ValueError('strict final12 reference matrix differs')
    if physical!=original:raise ValueError('native evaluation rows differ from immutable old audit')
    cps=[cp for cp in cell['checkpoints'] if cp['episode']==96]
    physical_cp=[cp for cp in parsed['checkpoints.jsonl'] if cp['episode']==96]
    if len(cps)!=1 or physical_cp!=cps:raise ValueError('reference final checkpoint ledger differs')
    cp=cps[0]
    if (Path(cp['path']).resolve()!=(home/'checkpoints/ep096.pt').resolve()
            or cp['method']!='main' or cp['seed']!=seed or cp['cumulative_updates']!=24
            or cp['policy_class']!='IntentAwareStreamSAGHMAPPOAgent'
            or cp['native_method']!='sa_ghmappo' or cp['checkpoint_contract']!='stream_main_cache_intent_v1'):
        raise ValueError('reference final checkpoint provenance differs')
    cp_raw,cp_ref=budget.read(cp['path'])
    if cp_ref['sha256']!=cp['sha256']:raise ValueError('reference final checkpoint SHA differs')
    payload=torch.load(io.BytesIO(cp_raw),map_location='cpu',weights_only=False)
    identity={k:v for k,v in cp.items() if k not in {'path','sha256','tensor_sha256'}}
    tensors=payload['state_dict']
    tensor_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in tensors.values())).hexdigest()
    if (payload.get('reward_curve_identity')!=identity or payload.get('options')!=cp['options']
            or payload.get('contract')!=cp['checkpoint_contract'] or tensor_sha!=cp['tensor_sha256']
            or int(tensors['completed_updates'])!=24):
        raise ValueError('reference final checkpoint serialized identity differs')
    expected={r['case_id']:r for r in original}
    states={cid:dict(frames=0,reward=0.,state_bytes=0.,state_link=0.,handoff=0,
                    completed=Counter(),delivered_by_frame=Counter(),native_completed_by_frame={},
                    frame_hash=hashlib.sha256(),control_hash=hashlib.sha256(),
                    events=set(),last_event_frame=-1,status_counts=Counter()) for cid in CASES}
    def target(row):
        if row['phase']!='evaluation' or row['episode']!=96:return False
        if (row['method']!='main' or row['seed']!=seed or row['case_id'] not in states
                or row['checkpoint_episode']!=96 or row['checkpoint_sha256']!=cp['sha256']):
            raise ValueError('reference final raw identity differs')
        return True
    receipt_path=home/'receipts.jsonl.gz'
    for row in budget.gzip_rows(receipt_path,complete['output_sha256'][receipt_path.name]):
        if not target(row):continue
        if row['record_kind']!='frame_receipt':raise ValueError('unexpected final frame record kind')
        cid=row['case_id'];state=states[cid];f=row['frame_index'];receipt=row['receipt']
        if (type(f) is not int or f!=state['frames'] or f>=expected[cid]['steps']
                or receipt['frame_index']!=f or receipt['reward_components']!=row['reward_components']):
            raise ValueError('reference frame gap/duplicate/native receipt differs')
        close(row['reward'],sum(receipt['reward_components'].values()),'native frame reward')
        state['frames']+=1;state['reward']+=row['reward']
        state['state_bytes']+=nonnegative(receipt['state_transfer_bytes'],'state bytes')
        state['state_link']+=nonnegative(receipt['state_transfer_link_frames'],'state link frames')
        node_count=nonnegative(receipt['reward_components']['completed_nodes'],'completed node reward component')
        if not float(node_count).is_integer():raise ValueError('native completed node count fractional')
        state['native_completed_by_frame'][f]=int(node_count)
        state['frame_hash'].update(packed({k:row[k] for k in ('frame_index','reward','reward_components')}).encode())
    control_path=home/'controls.jsonl.gz'
    for row in budget.gzip_rows(control_path,complete['output_sha256'][control_path.name]):
        if not target(row):continue
        if row['record_kind']=='decision_rollout':continue  # No legacy MC audit is repeated.
        if row['record_kind']!='native_control':raise ValueError('unexpected final control record kind')
        cid=row['case_id'];state=states[cid];event=row['event'];f=event['frame_index'];rid=event['request_id']
        if type(f) is not int or not 0<=f<state['frames'] or f<state['last_event_frame']:
            raise ValueError('reference native control frame invalid')
        request=cases[cid]['requests'].get(rid)
        if (request is None or event['vehicle_id']!=request['vehicle_id']
                or event['workflow_id']!=request['workflow_id'] or f<request['arrival_frame']):
            raise ValueError('native request identity not bound to original arrivals')
        if (f,event['vehicle_id']) in state['events']:
            raise ValueError('native event duplicated within physical vehicle/frame')
        if state['completed'][rid]>=request['total_nodes']:
            raise ValueError('native event after request completion')
        state['events'].add((f,event['vehicle_id']));state['last_event_frame']=f
        status=event['status']
        if status not in KNOWN_STATUSES:raise ValueError('unknown native delivery status: '+str(status))
        if type(event['service_target_changed']) is not bool or type(event['association_handoff']) is not bool:
            raise ValueError('native target-change/association fields not literal booleans')
        state['handoff']+=int(event['service_target_changed'] and status in {'state_restore','state_link_queue'})
        state['status_counts'][status]+=1
        nonnegative(event['state_transfer_link_frames'],'event state link frames')
        if status in DELIVERED_STATUSES:
            state['completed'][rid]+=1;state['delivered_by_frame'][f]+=1
        state['control_hash'].update(packed(event).encode())
    endpoints=[]
    for cid in CASES:
        state=states[cid];row=expected[cid];request_map=cases[cid]['requests']
        if (row['physical_terminal'] is not True or row['bootstrap_team_tail']!=0
                or not 0<row['steps']<=cases[cid]['horizon'] or state['frames']!=row['steps']
                or row['checkpoint_episode']!=96 or row['checkpoint_sha256']!=cp['sha256']
                or row['environment_revision']!='v3' or row['action_contract']!='joint_stream_control_v2'):
            raise ValueError('reference final complete episode/physical contract differs')
        close(state['reward'],row['reward'],'reference native return')
        close(state['state_bytes'],row['state_transfer_bytes'],'reference state bytes')
        close(state['state_link'],row['state_transfer_link_frames'],'reference state link frames')
        close(state['handoff'],row['handoff_failed_count'],'reference v3 handoff failure events')
        if any(state['delivered_by_frame'][f]!=count for f,count in state['native_completed_by_frame'].items()):
            raise ValueError('delivered-node statuses differ from native completion reward components')
        completed=sum(state['completed'][rid]==req['total_nodes'] for rid,req in request_map.items())
        close(completed,row['completed_requests'],'derived completed requests')
        close(sum(state['completed'].values()),row['completed_nodes'],'derived delivered nodes')
        if row['request_count']!=len(request_map) or row['released_request_ids']!=sorted(request_map):
            raise ValueError('reference request/release matrix differs')
        if max(req['arrival_frame'] for req in request_map.values())>=row['steps']:
            raise ValueError('final reference omits scheduled late arrivals')
        if row['steps']<cases[cid]['horizon'] and completed!=len(request_map):
            raise ValueError('early termination without all delivered requests')
        per_request={r['request_id']:r for r in row['per_request']}
        if len(row['per_request'])!=len(request_map) or set(per_request)!=set(request_map):
            raise ValueError('reference per-request matrix differs')
        for rid,req in request_map.items():
            q=per_request[rid]
            if (q['total_nodes']!=req['total_nodes'] or q['completed_nodes']!=state['completed'][rid]
                    or q['completed']!=(state['completed'][rid]==req['total_nodes'])):
                raise ValueError('reference per-request node/completion status differs')
        digest_key='evaluation|96|'+cid
        if (state['frame_hash'].hexdigest()!=cell['frame_digests'][digest_key]
                or state['control_hash'].hexdigest()!=cell['control_digests'][digest_key]):
            raise ValueError('reference final native frame/control digest differs from old audit')
        endpoints.append(dict(seed=seed,case_id=cid,episode=96,reward=row['reward'],
            completed_requests=completed,state_transfer_bytes=state['state_bytes'],
            state_transfer_link_frames=state['state_link'],handoff_failed_count=state['handoff'],
            checkpoint_sha256=cp['sha256'],frame_digest=state['frame_hash'].hexdigest(),
            control_digest=state['control_hash'].hexdigest(),physical_steps=state['frames'],
            delivered_node_count=sum(state['completed'].values()),
            completion_derivation='per_request_native_delivery_status_count_equals_frozen_workflow_node_count'))
    return endpoints,dict(seed=seed,complete=terminal_ref,manifest=manifest_ref,
        noncompressed_raw_files=raw_refs,final_checkpoint=cp_ref)


def qualify(campaign_path,audit_path,output,*,caps=None):
    campaign_path=Path(campaign_path);audit_path=Path(audit_path);output=Path(output)
    claim=Path(str(output)+'.claim.json')
    if output.exists() or claim.exists():raise FileExistsError(str(output))
    budget=ReadBudget(output,caps)
    # Authenticate the path owner before creating even a claim in an old run.
    campaign,campaign_ref=read_json(campaign_path,budget,CAMPAIGN_SHA)
    original_root=Path(campaign['root']).resolve();destination=output.resolve()
    protected=(original_root,campaign_path.resolve().parent,audit_path.resolve().parent)
    if any(destination==path or path in destination.parents for path in protected):
        raise ValueError('qualification output cannot modify an old scientific package')
    output.parent.mkdir(parents=True,exist_ok=True)
    budget.save(claim,dict(status='single_readonly_reference_qualification_claim',started_unix=time.time(),
        campaign_path=str(campaign_path.resolve()),audit_path=str(audit_path.resolve()),
        output=str(output.resolve()),science_interactions=0))
    try:
        audit,audit_ref=read_json(audit_path,budget,AUDIT_SHA)
        if (campaign['source_commit']!=SOURCE_COMMIT or campaign['training_case_ids']!=CASES
                or campaign['seeds']!=SEEDS or audit['campaign_sha256']!=CAMPAIGN_SHA
                or audit['source_commit']!=SOURCE_COMMIT or audit['status']!='curve_receipts_and_checkpoints_audited'
                or audit['final_selection']!='episode96_fixed_no_peak'):
            raise ValueError('frozen reward campaign/reference audit identity differs')
        environment=[]
        for name in ENV_SOURCES:
            _,ref=budget.read(ROOT/name,keep=False)
            if ref['sha256']!=campaign['source_fingerprints'][name]:
                raise ValueError('native delivery status/summary source changed')
            environment.append(ref)
        protocol,protocol_ref=read_json(campaign['protocol']['path'],budget,campaign['protocol']['sha256'])
        raw,pt_ref=budget.read(campaign['protocol']['protocol_pt'])
        if pt_ref['sha256']!=campaign['protocol']['protocol_pt_sha256']:
            raise ValueError('frozen protocol.pt identity differs')
        cases=protocol_cases(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False))
        cells=[c for c in audit['cells'] if c['method']=='main']
        if len(cells)!=5 or {c['seed'] for c in cells}!=set(SEEDS):
            raise ValueError('reference original five-seed main pool differs')
        endpoints=[];provenance=[]
        for seed in SEEDS:
            cell=next(c for c in cells if c['seed']==seed)
            home=Path(campaign['outputs']['main'][str(seed)])
            rows,refs=qualify_cell(home,cell,campaign,CAMPAIGN_SHA,budget,cases)
            endpoints.extend(rows);provenance.append(refs)
        if len(endpoints)!=60 or {(r['seed'],r['case_id']) for r in endpoints}!={(s,c) for s in SEEDS for c in CASES}:
            raise ValueError('reference strict final60 endpoint matrix incomplete')
        _,qualifier_source=budget.read(Path(__file__).resolve(),keep=False)
        result=dict(status='reference_final60_native_endpoints_qualified',reviewed_at=time.time(),
            campaign_sha256=CAMPAIGN_SHA,audit_sha256=AUDIT_SHA,source_commit=SOURCE_COMMIT,
            endpoints=endpoints,campaign=campaign_ref,audit=audit_ref,protocol=protocol_ref,
            protocol_pt=pt_ref,native_environment_sources=environment,cell_provenance=provenance,
            qualifier_source=qualifier_source,python_version=sys.version,
            raw_source_files=budget.files,
            bill=dict(total_source_bytes_read=budget.input_bytes,compressed_bytes_read=budget.compressed_bytes,
                decoded_bytes_read=budget.decoded_bytes,decoded_conservative_inflight_upper_bound=65536,
                elapsed_seconds=time.monotonic()-budget.started,limits=budget.caps,
                new_physical_frames=0,forward_calls=0,training=0,fitting=0,downloads=0,new_indices=0),
            derivation_boundaries=[
                'Only old main final60 native endpoints are qualified; old20-cell reward/MC audit is not rerun.',
                'Reward remains the immutable old audit value and is cross-checked against selected final native receipts.',
                'Nonnegative enforcement applies to physical costs and completion counts; signed native reward components/returns are preserved and are not censored.',
                'Legacy reduced controls omit node_id and workflow_completed; completion is derived from per-request node_completed/vehicle_fallback_completed counts and frozen workflow node totals, with per-frame native completed-node component consistency. No direct workflow flag or node-level identity observation is claimed.',
                'V3 handoff_failed_count counts service_target_changed && status in {state_restore,state_link_queue}; not association_handoff or a unique request failure rate.',
                'Reading skipped nonfinal gzip rows binds source SHA/accounting only; no nonfinal rewards/MC/optimization are re-audited.',
                'Consumed training data, no new independent/generalization evidence. All old files and failures remain unchanged.'],
            paper_goal_complete=False)
        budget.save(output,result)
        return result
    except BaseException as error:
        failure=Path(str(output)+'.failure.json')
        if not failure.exists():
            with failure.open('x') as stream:
                stream.write(packed(dict(status='reference_endpoint_qualification_failed_preserved',
                    error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()[-32000:],
                    total_source_bytes_read=budget.input_bytes,compressed_bytes_read=budget.compressed_bytes,
                    decoded_bytes_read=budget.decoded_bytes,elapsed_seconds=time.monotonic()-budget.started,
                    science_interactions=0,paper_goal_complete=False)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(qualify(args.campaign,args.audit,args.output)))
