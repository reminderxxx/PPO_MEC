"""Synthetic-only, zero-simulator acceptance for final60 endpoint qualification."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

import scripts.qualify_stream_reference_endpoints as q


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(q.packed(value))
def write_rows(path,rows):Path(path).write_text(''.join(q.packed(r) for r in rows))
def gzwrite(path,rows):
    with gzip.open(path,'wt') as stream:
        for row in rows:stream.write(q.packed(row))
def gzread(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def synthetic_frame_rows(seed,cid,checkpoint):
    receipts=[];controls=[]
    schedule=[ [('v','v/first','first','state_restore',True)],
               [('v','v/first','first','node_completed',False),('w','w/second','second','vehicle_compute',False)],
               [('v','v/first','first','node_completed',False),('w','w/second','second','waiting',False)],
               [('w','w/second','second','vehicle_fallback_completed',False)] ]
    frames_hash=hashlib.sha256();control_hash=hashlib.sha256()
    for f,events in enumerate(schedule):
        delivered=sum(e[3] in q.DELIVERED_STATUSES for e in events)
        parts=dict(completed_nodes=float(delivered),state_transfer=-.01 if f==0 else 0.)
        reward=sum(parts.values())
        identity=dict(method='main',seed=seed,case_id=cid,phase='evaluation',episode=96,
                      checkpoint_episode=96,checkpoint_sha256=checkpoint)
        body=dict(frame_index=f,reward=reward,reward_components=parts,
            receipt=dict(frame_index=f,reward_components=parts,state_transfer_bytes=5 if f==0 else 0,
                         state_transfer_link_frames=1 if f==0 else 0))
        receipts.append(dict(identity,record_kind='frame_receipt',**body))
        frames_hash.update(q.packed({name:body[name] for name in ('frame_index','reward','reward_components')}).encode())
        for vehicle,rid,wid,status,changed in events:
            event=dict(frame_index=f,vehicle_id=vehicle,request_id=rid,workflow_id=wid,status=status,
                control={'execution_target':'a'},control_planning={},cache_request={},
                state_transfer_link_frames=1 if f==0 else 0,execution_rsu_id='a',
                service_target_changed=changed,association_handoff=False)
            controls.append(dict(identity,record_kind='native_control',event=event))
            control_hash.update(q.packed(event).encode())
    return receipts,controls,frames_hash.hexdigest(),control_hash.hexdigest()


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    root=tmp_path_factory.mktemp('reference_endpoints_software_fixture');code=root/'code'
    source_fingerprints={}
    for name in q.ENV_SOURCES:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# Explicit synthetic frozen environment identity; no simulator import or execution.\n')
        source_fingerprints[name]=sha(path)
    source=root/'source.json';write(source,{'synthetic_software_fixture':True})
    arrivals=[SimpleNamespace(request_id='v/first',vehicle_id='v',frame=0,
                  workflow=SimpleNamespace(workflow_id='first',nodes=[1,2])),
              SimpleNamespace(request_id='w/second',vehicle_id='w',frame=1,
                  workflow=SimpleNamespace(workflow_id='second',nodes=[1]))]
    data=dict(cases={'train':[dict(case_id=c,session=0,arrivals=deepcopy(arrivals)) for c in q.CASES]},
              sessions=[dict(split='train',frames=[{}]*4)])
    pt=root/'protocol.pt';torch.save(data,pt)
    campaign=dict(source_commit=q.SOURCE_COMMIT,seeds=q.SEEDS,training_case_ids=q.CASES,
        root=str(root),
        source_fingerprints=source_fingerprints,outputs={'main':{str(s):str(root/f'main{s}') for s in q.SEEDS}},
        protocol=dict(path=str(source),sha256=sha(source),protocol_pt=str(pt),protocol_pt_sha256=sha(pt)))
    campaign_path=root/'campaign.json';write(campaign_path,campaign)
    cells=[]
    for seed in q.SEEDS:
        home=root/f'main{seed}';(home/'checkpoints').mkdir(parents=True)
        tensors=dict(completed_updates=torch.tensor(24),weight=torch.tensor([1.]))
        cp=dict(episode=96,method='main',seed=seed,cumulative_updates=24,
            policy_class='IntentAwareStreamSAGHMAPPOAgent',native_method='sa_ghmappo',
            checkpoint_contract='stream_main_cache_intent_v1',options={'method':'sa_ghmappo'},
            path=str(home/'checkpoints/ep096.pt'))
        payload=dict(state_dict=tensors,options=cp['options'],contract=cp['checkpoint_contract'],
                     reward_curve_identity={k:v for k,v in cp.items() if k!='path'})
        torch.save(payload,cp['path']);cp['sha256']=sha(cp['path'])
        cp['tensor_sha256']=hashlib.sha256(b''.join(v.numpy().tobytes() for v in tensors.values())).hexdigest()
        evaluation=[];receipts=[];controls=[];frame_digests={};control_digests={}
        for cid in q.CASES:
            frames,events,fh,ch=synthetic_frame_rows(seed,cid,cp['sha256'])
            receipts.extend(frames);controls.extend(events)
            frame_digests['evaluation|96|'+cid]=fh;control_digests['evaluation|96|'+cid]=ch
            evaluation.append(dict(method='main',seed=seed,case_id=cid,episode=96,
                checkpoint_episode=96,checkpoint_sha256=cp['sha256'],physical_terminal=True,
                bootstrap_team_tail=0,steps=4,environment_revision='v3',action_contract='joint_stream_control_v2',
                reward=2.99,state_transfer_bytes=5,state_transfer_link_frames=1,handoff_failed_count=1,
                completed_requests=2,completed_nodes=3,request_count=2,
                released_request_ids=['v/first','w/second'],
                per_request=[dict(request_id='v/first',total_nodes=2,completed_nodes=2,completed=True),
                             dict(request_id='w/second',total_nodes=1,completed_nodes=1,completed=True)]))
        # A skipped nonfinal row has no endpoint qualification: its intentional
        # bogus status must not make this task repeat the old training audit.
        controls.insert(0,dict(phase='train',episode=1,event={'status':'not_reaudited'}))
        write(home/'protocol.json',dict(plan=campaign,campaign_sha256=sha(campaign_path)))
        write_rows(home/'evaluation.jsonl',evaluation);write_rows(home/'checkpoints.jsonl',[cp])
        write_rows(home/'training.jsonl',[{'synthetic_only':True}]);write_rows(home/'updates.jsonl',[{'synthetic_only':True}])
        gzwrite(home/'receipts.jsonl.gz',receipts);gzwrite(home/'controls.jsonl.gz',controls)
        complete=dict(status='complete',method='main',seed=seed,campaign_sha256=sha(campaign_path),
            source_commit=q.SOURCE_COMMIT,output_sha256={name:sha(home/name) for name in
                ['protocol.json','training.jsonl','evaluation.jsonl','updates.jsonl','checkpoints.jsonl',
                 'receipts.jsonl.gz','controls.jsonl.gz']})
        write(home/'complete.json',complete);write(home/'manifest.json',complete)
        cells.append(dict(method='main',seed=seed,evaluation=evaluation,checkpoints=[cp],
                          frame_digests=frame_digests,control_digests=control_digests))
    audit=dict(source_commit=q.SOURCE_COMMIT,campaign_sha256=sha(campaign_path),
        status='curve_receipts_and_checkpoints_audited',final_selection='episode96_fixed_no_peak',cells=cells)
    audit_path=root/'audit.json';write(audit_path,audit)
    return SimpleNamespace(root=root,code=code,campaign=campaign_path,audit=audit_path,
        campaign_sha=sha(campaign_path),audit_sha=sha(audit_path),
        originals={p:p.read_bytes() for p in root.rglob('*') if p.is_file()})


@contextmanager
def patched(package):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q,'ROOT',package.code);patch.setattr(q,'CAMPAIGN_SHA',package.campaign_sha)
        patch.setattr(q,'AUDIT_SHA',package.audit_sha)
        try:yield
        finally:
            for p,b in package.originals.items():
                if p.read_bytes()!=b:p.write_bytes(b)


def refresh(home,name):
    row=json.loads((home/'complete.json').read_text());row['output_sha256'][name]=sha(home/name)
    write(home/'complete.json',row);write(home/'manifest.json',row)


def test_exact60_qualified_without_any_simulator_or_forward(package,tmp_path):
    with patched(package):
        out=tmp_path/'qualification.json';r=q.qualify(package.campaign,package.audit,out)
        assert r['status']=='reference_final60_native_endpoints_qualified'
        assert len(r['endpoints'])==60 and {(x['seed'],x['case_id']) for x in r['endpoints']}=={(s,c) for s in q.SEEDS for c in q.CASES}
        assert all(x['completed_requests']==2 and x['handoff_failed_count']==1 for x in r['endpoints'])
        assert all(x['state_transfer_bytes']==5 and x['state_transfer_link_frames']==1 for x in r['endpoints'])
        assert r['bill']['new_physical_frames']==r['bill']['forward_calls']==r['bill']['fitting']==0
        assert r['bill']['compressed_bytes_read']>0 and r['bill']['decoded_bytes_read']>0
        assert r['bill']['total_source_bytes_read']>=r['bill']['compressed_bytes_read']
        assert any('omit node_id' in x for x in r['derivation_boundaries'])
        assert out.is_file()
        with pytest.raises(FileExistsError):q.qualify(package.campaign,package.audit,out)


@pytest.mark.parametrize('kind',['receipt_negative','receipt_cost','receipt_reward','frame_gap','frame_missing',
    'unknown_status','wrong_request','duplicate_event','missing_completion','too_many_completions',
    'association_confusion','target_bool','post_complete_event','wrong_control_hash'])
def test_native_endpoint_tampering_rejected_after_producer_sha_refresh(package,tmp_path,kind):
    with patched(package):
        home=package.root/'main7'
        if kind.startswith('receipt') or kind.startswith('frame'):
            path=home/'receipts.jsonl.gz';rows=gzread(path)
            if kind=='receipt_negative':rows[0]['receipt']['state_transfer_bytes']=-1
            elif kind=='receipt_cost':rows[0]['receipt']['state_transfer_bytes']+=1
            elif kind=='receipt_reward':rows[0]['reward']+=1
            elif kind=='frame_gap':rows[0]['frame_index']=1
            else:rows.pop(0)
        else:
            path=home/'controls.jsonl.gz';rows=gzread(path)
            indices=[i for i,x in enumerate(rows) if x.get('phase')=='evaluation'];i=indices[0]
            if kind=='unknown_status':rows[i]['event']['status']='unknown'
            elif kind=='wrong_request':rows[i]['event']['request_id']='missing/request'
            elif kind=='duplicate_event':rows.insert(i,deepcopy(rows[i]))
            elif kind=='missing_completion':next(x for x in rows if x.get('event',{}).get('status')=='node_completed')['event']['status']='waiting'
            elif kind=='too_many_completions':rows[i]['event']['status']='node_completed'
            elif kind=='association_confusion':
                rows[i]['event']['service_target_changed']=False;rows[i]['event']['association_handoff']=True
            elif kind=='target_bool':rows[i]['event']['service_target_changed']=1
            elif kind=='post_complete_event':
                old=deepcopy(rows[indices[3]]);old['event']['frame_index']=3;rows.insert(indices[-1]+1,old)
            elif kind=='wrong_control_hash':rows[i]['event']['control']['execution_target']='b'
        gzwrite(path,rows);refresh(home,path.name)
        out=tmp_path/'failed.json'
        with pytest.raises((ValueError,KeyError)):q.qualify(package.campaign,package.audit,out)
        assert not out.exists() and Path(str(out)+'.failure.json').is_file()
        with pytest.raises(FileExistsError):q.qualify(package.campaign,package.audit,out)


@pytest.mark.parametrize('kind',['old_audit','protocol','cp_bytes','cp_ledger','manifest','missing_final','wrong_endpoint_summary'])
def test_reference_hash_checkpoint_and_manifest_fail_closed(package,tmp_path,kind):
    with patched(package):
        home=package.root/'main7'
        if kind=='old_audit':package.audit.write_text(package.audit.read_text()+' ')
        elif kind=='protocol':(package.root/'source.json').write_text('{}')
        elif kind=='cp_bytes':(home/'checkpoints/ep096.pt').write_bytes(b'changed')
        elif kind=='cp_ledger':
            p=home/'checkpoints.jsonl';row=json.loads(p.read_text());row['cumulative_updates']=25;write_rows(p,[row]);refresh(home,p.name)
        elif kind=='manifest':write(home/'manifest.json',{'status':'wrong'})
        elif kind in ['missing_final','wrong_endpoint_summary']:
            p=home/'evaluation.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
            if kind=='missing_final':rows.pop()
            else:rows[0]['handoff_failed_count']=0
            write_rows(p,rows);refresh(home,p.name)
        with pytest.raises((ValueError,KeyError)):q.qualify(package.campaign,package.audit,tmp_path/'qualified.json')


def test_read_budget_limits_are_enforced_before_accepting_more_bytes(tmp_path):
    path=tmp_path/'input';path.write_bytes(b'12345')
    b=q.ReadBudget(tmp_path/'out',dict(q.CAPS,maximum_input_bytes=4))
    with pytest.raises(RuntimeError,match='read budget'):b.read(path)
    assert b.input_bytes==4
    b=q.ReadBudget(tmp_path/'out',dict(q.CAPS,maximum_wall_seconds=1));b.started=time.monotonic()-1
    with pytest.raises(RuntimeError,match='wall'):b.read(path)


def test_decoded_and_output_caps_keep_failed_sources_and_never_overwrite(tmp_path):
    path=tmp_path/'rows.gz';gzwrite(path,[{'long':'x'*100}])
    b=q.ReadBudget(tmp_path/'out',dict(q.CAPS,maximum_decoded_bytes=65540))
    with pytest.raises(RuntimeError,match='decoded'):list(b.gzip_rows(path,sha(path)))
    assert b.decoded_bytes+65536<=65540
    b=q.ReadBudget(tmp_path/'out',dict(q.CAPS,maximum_output_bytes=65540))
    with pytest.raises(RuntimeError,match='output'):b.save(tmp_path/'out',{'value':'too big'})
    assert not (tmp_path/'out').exists()


def test_unknown_or_ambiguous_input_request_cardinality_is_not_invented():
    a=SimpleNamespace(request_id='v/a',vehicle_id='v',frame=0,
                      workflow=SimpleNamespace(workflow_id='a',nodes=[]))
    data=dict(cases={'train':[dict(case_id=c,session=0,arrivals=[a]) for c in q.CASES]},
              sessions=[dict(split='train',frames=[{}])])
    with pytest.raises(ValueError,match='node count'):q.protocol_cases(data)
    data['cases']['train'].pop()
    with pytest.raises(ValueError,match='matrix'):q.protocol_cases(data)


def test_qualification_cannot_write_new_claim_inside_old_package(package):
    with patched(package):
        target=package.root/'new_inside_old.json'
        with pytest.raises(ValueError,match='cannot modify'):q.qualify(package.campaign,package.audit,target)
        assert not target.exists() and not Path(str(target)+'.claim.json').exists()
