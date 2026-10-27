"""Prospective synthetic qualification; no external request or policy rollout."""
import csv
import io
import json
from pathlib import Path
import urllib.parse
import hashlib

import pytest

from scripts.prepare_tgsim_input import BudgetLimit, metadata_pin
from scripts.prepare_tgsim_paged_input import (PagedBudget, NoRedirect, acquire,
    append_page, disk_bytes, page_url, parse_count, system_id_key, validate_plan)


class Response(io.BytesIO):
    status = 200
    def __init__(self, url, body):
        super().__init__(body);self.url=url
        self.headers={'Content-Type':'text/csv','Content-Length':str(len(body))}
    def geturl(self):return self.url


class FixtureTransport:
    def __init__(self, plan, *, short=False, duplicate=False, count_drift=False,
                 version_drift=False, after_drift=False, failure=False):
        self.plan=plan;self.calls=[];self.seen={}
        self.short=short;self.duplicate=duplicate;self.count_drift=count_drift
        self.version_drift=version_drift;self.after_drift=after_drift;self.failure=failure

    def open(self, request, **kwargs):
        url=request.full_url;self.calls.append(url);self.seen[url]=self.seen.get(url,0)+1
        source=next(s for s in self.plan['sources'] if s['asset_id'] in url)
        if '/api/views/' in url:
            data=metadata(source)
            if self.version_drift or (self.after_drift and self.seen[url]==2):data['rowsUpdatedAt']=2
            body=json.dumps(data).encode()
        elif '.json?' in url:
            body=json.dumps([{'count':'4' if self.count_drift else '3'}]).encode()
        elif '/api/assets/' in url:
            body=b'x,y\n1,2\n'
        else:
            if self.failure:raise OSError('synthetic network interruption')
            offset=int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)['$offset'][0])
            ids=[1,2] if offset==0 else [3]
            if offset and self.short:ids=[]
            if offset and self.duplicate:ids=[2]
            out=io.StringIO();writer=csv.writer(out)
            writer.writerow(['source_row_id']+source['expected_native_fields'])
            for sid in ids:writer.writerow([sid,1,f'0.{sid-1}',1,2,1])
            body=out.getvalue().encode()
        return Response(url,body)


def metadata(source):
    return dict(id=source['asset_id'],rowsUpdatedAt=1,
                columns=[{'name':n,'fieldName':n,'dataTypeName':'number'} for n in source['expected_native_fields']],
                metadata={'attachments':[dict(assetId=a['asset_id'],name=a['name']) for a in source['support_csv_attachments']]})


def setup(tmp_path,geometry=False):
    raw=tmp_path/'raw';analysis=tmp_path/'analysis';raw.mkdir();analysis.mkdir()
    sources=[]
    for asset in ('synthetic-a','synthetic-b'):
        source=dict(asset_id=asset,expected_native_fields=['id','time','xloc_kf','yloc_kf','run_index'],
                    expected_count_from_frozen_metadata_not_fresh_query=3,support_csv_attachments=[],
                    trajectory_endpoint=f'https://data.transportation.gov/resource/{asset}.csv',
                    metadata_url=f'https://data.transportation.gov/api/views/{asset}.json',
                    row_count_url=f'https://data.transportation.gov/resource/{asset}.json?count',
                    fixed_query={'$select':':id as source_row_id,*','$order':':id','$limit':2,'$offset':0})
        if geometry:source['support_csv_attachments']=[dict(asset_id=f'{asset}-geometry-{n}',name=f'geometry_{n}.csv') for n in range(8)]
        source['metadata_pin']=metadata_pin(metadata(source));sources.append(source)
    plan=dict(sources=sources,page_rows=2,maximum_trajectory_pages=4,maximum_http_requests=12,
              max_download_bytes_new_task=100000,max_cumulative_new_disk_bytes_including_old_failure=32*1024**2,
              max_wall_clock_seconds=30,max_rows_to_index=6)
    if geometry:plan['maximum_http_requests']=28
    return plan,raw,analysis,PagedBudget(plan,[raw,analysis])


def test_complete_composition_is_native_derived_and_counts_once(tmp_path):
    plan,raw,analysis,budget=setup(tmp_path);opener=FixtureTransport(plan)
    result=acquire(plan,raw,analysis,budget,opener=opener)
    assert budget.requests==len(opener.calls)==12 and budget.rows==6
    assert len(result['page_inventory'])==4
    assert all(c['rows']==3 and c['derived_from_pages'] and not c['bulk_byte_equality_claim'] for c in result['trajectory_checks'])
    assert all(c['geometry_gate'].startswith('UNQUALIFIED') for c in result['trajectory_checks'])
    assert len((raw/'synthetic-a_derived_native.csv').read_text().splitlines())==4
    assert all(n==1 if '.csv?' in u else n==2 for u,n in opener.seen.items())


@pytest.mark.parametrize('mode,match,requests',[
    ('short','missing/short',6),('duplicate','duplicate or out-of-order',6),
    ('count_drift','fresh source row count',2),('version_drift','metadata version',1),
    ('after_drift','metadata version',9),('failure','synthetic network',5)])
def test_any_failure_stops_globally_and_preserves_without_retry(tmp_path,mode,match,requests):
    plan,raw,analysis,budget=setup(tmp_path);opener=FixtureTransport(plan,**{mode:True})
    with pytest.raises((ValueError,OSError),match=match):acquire(plan,raw,analysis,budget,opener=opener)
    assert len(opener.calls)==requests and budget.requests==requests
    assert all(n<=1 for u,n in opener.seen.items() if '.csv?' in u)
    if mode=='failure':
        saved=json.loads((analysis/'synthetic-a_page_000_receipt.json').read_text())
        assert saved['status']=='failed' and (raw/'synthetic-a_page_000.csv').exists()


def test_count_schema_and_fractional_or_negative_values_rejected(tmp_path):
    path=tmp_path/'count.json'
    for body in ([{'count':'3.0'}],[{'count':'-1'}],[{'count':'3','extra':1}],[],{'count':3}):
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError):parse_count(path,3)


def test_source_order_and_page_url_are_frozen(tmp_path):
    plan,*_=setup(tmp_path);url=page_url(plan['sources'][0],50000,50000)
    query=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query=={'$select':[':id as source_row_id,*'],'$order':[':id'],'$limit':['50000'],'$offset':['50000']}
    assert system_id_key('10')>system_id_key('2')
    assert system_id_key('row-abc')<system_id_key('row-abd')
    with pytest.raises(ValueError):system_id_key('unrecognized')
    with pytest.raises(ValueError):NoRedirect().redirect_request(None,None,302,'',{},'https://data.transportation.gov/elsewhere')


def test_missing_header_extra_rows_and_id_order_rejected(tmp_path):
    plan,raw,analysis,budget=setup(tmp_path);source=plan['sources'][0];path=raw/'fixture.csv'
    for content,match in [('wrong\n','header'),
        ('source_row_id,id,time,xloc_kf,yloc_kf,run_index\n2,1,0,1,2,1\n1,1,0.1,1,2,1\n','out-of-order'),
        ('source_row_id,id,time,xloc_kf,yloc_kf,run_index\n1,1,0,1,2,1\n2,1,0.1,1,2,1\n','more than')]:
        path.write_text(content)
        with pytest.raises(ValueError,match=match):
            append_page(path,source,csv.writer(io.StringIO()),1,None,budget)


def test_request_and_page_caps_stop_before_next_request(tmp_path):
    plan,raw,analysis,budget=setup(tmp_path);budget.maximum_requests=1;opener=FixtureTransport(plan)
    with pytest.raises(BudgetLimit,match='HTTP request'):acquire(plan,raw,analysis,budget,opener=opener)
    assert len(opener.calls)==1
    with pytest.raises(ValueError,match='retried'):budget.request(opener.calls[0])


def test_disk_includes_old_failure_and_cannot_free_it_by_reset(tmp_path):
    plan,raw,analysis,_=setup(tmp_path);old=tmp_path/'old';old.mkdir();(old/'preserved').write_bytes(b'old')
    budget=PagedBudget(plan,[old,raw,analysis]);budget.proposal['max_new_disk_bytes']=8*1024**2+2
    with pytest.raises(BudgetLimit,match='disk'):budget.guard()
    assert (old/'preserved').read_bytes()==b'old' and disk_bytes([old,raw,analysis])==3
    with pytest.raises(ValueError,match='overlap'):disk_bytes([tmp_path,old])


def test_earlier_bulk_grant_cannot_authorize_new_plan(tmp_path):
    proposal=tmp_path/'proposal.json';proposal.write_text('{}')
    with pytest.raises(ValueError,match='newly approved SHA'):validate_plan(proposal,tmp_path/'old_grant.json')


def test_independent_consumer_checks_failure_files_and_exact_bill(tmp_path,monkeypatch):
    from scripts import audit_tgsim_paged_input as consumer
    plan,raw,analysis,budget=setup(tmp_path);plan['destination_raw_directory']=str(raw)
    opener=FixtureTransport(plan,failure=True)
    with pytest.raises(OSError):acquire(plan,raw,analysis,budget,opener=opener)
    proposal=tmp_path/'frozen_proposal.json';proposal.write_text(json.dumps(plan))
    grant=tmp_path/'grant.json';grant.write_text('synthetic software grant')
    sha=hashlib.sha256(proposal.read_bytes()).hexdigest();monkeypatch.setattr(consumer,'PROPOSAL_SHA',sha)
    record=dict(proposal_sha256=sha,source_commit='synthetic',grant_sha256=hashlib.sha256(grant.read_bytes()).hexdigest(),
                training_frames=0,evaluation_frames=0,hidden_openings=0,
                command_arguments=dict(proposal=str(proposal),grant=str(grant)))
    (analysis/'execution_claim.json').write_text(json.dumps(dict(record,proposal=plan)))
    completion=dict(record,status='failed_preserved_do_not_retry',http_requests=budget.requests,
                    total_body_download_bytes=budget.download_bytes,total_body_download_bytes_upper_bound=budget.download_bytes,
                    two_task_body_upper_bound=consumer.OLD_BODY_BOUND+budget.download_bytes,cumulative_disk_bytes=disk_bytes([raw,analysis]),
                    cumulative_disk_roots=[str(raw),str(analysis)],raw_files=[dict(path=p.name,bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in raw.iterdir()])
    path=analysis/'completion.json';path.write_text(json.dumps(completion))
    result=consumer.audit(analysis)
    assert not result['acquisition_complete'] and len(result['request_receipts'])==5
    completion['two_task_body_upper_bound']-=1;path.write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='bill mismatch'):consumer.audit(analysis)
    completion['two_task_body_upper_bound']+=1;path.write_text(json.dumps(completion))
    (raw/'synthetic-a_page_000.csv').write_text('tampered')
    with pytest.raises(ValueError,match='bytes/hash'):consumer.audit(analysis)


def test_independent_success_consumer_rejects_missing_page_inventory(tmp_path,monkeypatch):
    from scripts import audit_tgsim_paged_input as consumer
    plan,raw,analysis,budget=setup(tmp_path,geometry=True);plan['destination_raw_directory']=str(raw)
    result=acquire(plan,raw,analysis,budget,opener=FixtureTransport(plan))
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps(plan))
    grant=tmp_path/'grant.json';grant.write_text('synthetic')
    sha=hashlib.sha256(proposal.read_bytes()).hexdigest();monkeypatch.setattr(consumer,'PROPOSAL_SHA',sha)
    record=dict(proposal_sha256=sha,source_commit='synthetic',grant_sha256=hashlib.sha256(grant.read_bytes()).hexdigest(),
                training_frames=0,evaluation_frames=0,hidden_openings=0,
                command_arguments=dict(proposal=str(proposal),grant=str(grant)))
    (analysis/'execution_claim.json').write_text(json.dumps(dict(record,proposal=plan)))
    completion=dict(record,**result,status='acquired_and_input_checked_not_science_qualified',http_requests=budget.requests,
                    rows_indexed=budget.rows,total_body_download_bytes=budget.download_bytes,
                    total_body_download_bytes_upper_bound=budget.download_bytes,
                    two_task_body_upper_bound=consumer.OLD_BODY_BOUND+budget.download_bytes,cumulative_disk_bytes=disk_bytes([raw,analysis]),
                    cumulative_disk_roots=[str(raw),str(analysis)],raw_files=[dict(path=p.name,bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in raw.iterdir()])
    path=analysis/'completion.json';path.write_text(json.dumps(completion))
    assert consumer.audit(analysis)['native_indexed_rows_verified']==6
    completion['page_inventory'].pop();path.write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='page inventory'):consumer.audit(analysis)
