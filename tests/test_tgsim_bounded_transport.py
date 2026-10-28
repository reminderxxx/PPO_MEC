"""Synthetic transport/ancestor adversaries; no external HTTP or policy rollout."""
import copy
import http.client
import io
import json
from pathlib import Path

import pytest

from scripts.prepare_tgsim_input import BudgetLimit, InputDeadline, digest, save_new
from scripts.prepare_tgsim_paged_input import PagedBudget
from scripts.tgsim_bounded_transport import BoundedPageFetcher, ancestor_bill
from scripts.audit_tgsim_paged_input import audit_attempts

URL='https://data.transportation.gov/resource/fixture.csv?fixed'


class Response(io.BytesIO):
    status=200
    def __init__(self,body=b'complete'):
        super().__init__(body);self.headers={'Content-Type':'text/csv','Content-Length':str(len(body))}
    def geturl(self):return URL


class Broken(Response):
    def __init__(self,error=None):
        super().__init__(b'abc');self.headers.pop('Content-Length');self.n=0
        self.error=error or http.client.IncompleteRead(b'')
    def read1(self,size):
        self.n+=1
        if self.n==1:return super().read1(min(3,size))
        raise self.error


class Sequence:
    def __init__(self,*responses):self.responses=list(responses);self.calls=0
    def open(self,*args,**kwargs):
        self.calls+=1;return self.responses.pop(0)


def fixture(tmp_path,**changes):
    raw=tmp_path/'raw';raw.mkdir();analysis=tmp_path/'analysis';analysis.mkdir()
    plan=dict(automatic_retry=True,max_total_attempts_per_trajectory_page=3,max_extra_attempts_global=8,
              max_wall_clock_seconds=30,max_cumulative_new_disk_bytes_including_old_failure=32*1024**2,
              max_download_bytes_new_task=1000000,max_rows_to_index=10,maximum_http_requests=10)
    plan.update(changes);budget=PagedBudget(plan,[raw,analysis]);fetcher=BoundedPageFetcher(plan,analysis)
    path=raw/'fixture_page_000.csv.body';receipt=analysis/'fixture_page_000_receipt.json'
    return plan,raw,analysis,budget,fetcher,path,receipt


def test_interrupted_then_success_retains_every_attempt_and_bills_once(tmp_path):
    plan,raw,analysis,budget,fetcher,path,receipt=fixture(tmp_path)
    opener=Sequence(Broken(),Response())
    fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    logical=json.loads(receipt.read_text())
    assert opener.calls==budget.requests==2 and logical['selected_attempt']==2
    assert path.read_bytes()==b'complete' and Path(str(path)+'.attempt_1').read_bytes()==b'abc'
    assert budget.download_bytes==11 and budget.interrupted_body_upper_bound==65536
    b,u,n,_=audit_attempts(logical,receipt,raw,analysis,plan)
    assert (b,u,n)==(11,65536,2) and fetcher.extra_attempts==1
    # Altering either a failed original or the aggregate is detectable.
    logical['network_body_bytes']-=1
    with pytest.raises(ValueError,match='aggregate'):audit_attempts(logical,receipt,raw,analysis,plan)
    logical['network_body_bytes']+=1
    Path(str(path)+'.attempt_1').write_bytes(b'changed')
    with pytest.raises(ValueError,match='body/URL/hash'):audit_attempts(logical,receipt,raw,analysis,plan)


def test_three_transport_failures_exhaust_page_limit(tmp_path):
    plan,raw,analysis,budget,fetcher,path,receipt=fixture(tmp_path)
    opener=Sequence(Broken(),Broken(),Broken(),Response())
    with pytest.raises(http.client.IncompleteRead):fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    logical=json.loads(receipt.read_text());assert logical['selected_attempt'] is None
    assert opener.calls==3 and budget.requests==3 and not path.exists()
    assert len(list(raw.iterdir()))==3 and fetcher.extra_attempts==2
    assert audit_attempts(logical,receipt,raw,analysis,plan)[:3]==(9,196608,3)


@pytest.mark.parametrize('kind',['http','html','deadline','budget','unknown'])
def test_semantic_and_total_budget_errors_never_retry(tmp_path,kind):
    _,_,_,budget,fetcher,path,receipt=fixture(tmp_path)
    response=Response()
    if kind=='http':response.status=429
    if kind=='html':response.headers['Content-Type']='text/html'
    if kind=='deadline':response=Broken(InputDeadline('total deadline'))
    if kind=='budget':response=Broken(BudgetLimit('total cap'))
    if kind=='unknown':response=Broken(OSError('unknown transport diagnosis'))
    opener=Sequence(response,Response())
    with pytest.raises((ValueError,OSError,BudgetLimit)):
        fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    assert opener.calls==budget.requests==1 and fetcher.extra_attempts==0


@pytest.mark.parametrize('cap',['request','extra','bytes','time','disk'])
def test_global_caps_stop_before_another_request(tmp_path,cap):
    _,_,_,budget,fetcher,path,receipt=fixture(tmp_path)
    if cap=='request':budget.maximum_requests=1
    if cap=='extra':fetcher.plan['max_extra_attempts_global']=0
    if cap=='bytes':budget.proposal['max_download_bytes']=10
    if cap=='time':budget.clock=lambda:budget.started+30
    if cap=='disk':budget.proposal['max_new_disk_bytes']=8*1024**2-1
    opener=Sequence(Broken(),Response())
    with pytest.raises((BudgetLimit,InputDeadline)):
        fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    assert opener.calls==(0 if cap in ('time','disk') else 1)
    assert budget.download_bytes+budget.interrupted_body_upper_bound<=budget.proposal['max_download_bytes']


def test_old_uncertain_read_is_deducted_from_next_actual_read(tmp_path):
    _,_,_,budget,fetcher,path,receipt=fixture(tmp_path)
    budget.proposal['max_download_bytes']=10;budget.interrupted_body_upper_bound=8
    opener=Sequence(Response(b'abc'))
    with pytest.raises(BudgetLimit,match='declared body'):
        fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    assert budget.download_bytes==0 and budget.interrupted_body_upper_bound==8


def lineage_fixture(tmp_path,monkeypatch):
    from scripts import tgsim_bounded_transport as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'PRIVATE',tmp_path)
    root=dict(module.ORIGINAL_CAPS,sources=[],disk_roots=[],prior_accounted_body_bound=187568036)
    bindings=[];parent=root
    for index,(body,http,seconds,decoded) in enumerate([(35296468,25,383.804718084,122302692),(491850,5,23.285682834,122302692)]):
        directory=tmp_path/f'parent{index}';directory.mkdir();planpath=directory/'plan.json';save_new(planpath,parent)
        claim=dict(proposal=parent,proposal_sha256=digest(planpath),source_commit='fixture',training_frames=0,evaluation_frames=0,hidden_openings=0)
        save_new(directory/'execution_claim.json',claim)
        completion=dict(claim);completion.pop('proposal')
        completion.update(status='failed_preserved_do_not_retry',rows_indexed=0,command_arguments={'proposal':str(planpath)},
                          total_body_download_bytes_upper_bound=body,http_requests=http,elapsed_seconds=seconds,
                          decoded_csv_bytes=decoded,two_task_body_upper_bound=parent['prior_accounted_body_bound']+body)
        path=directory/'completion.json';save_new(path,completion)
        binding=dict(path=str(path),sha256=digest(path),plan_sha256=digest(planpath));bindings.insert(0,binding)
        parent=dict(root,prior_accounted_body_bound=completion['two_task_body_upper_bound'],
                    continuation_parent_completion=str(path),continuation_parent_completion_sha256=digest(path),
                    continuation_parent_plan_sha256=digest(planpath))
        parent.update(max_download_bytes_new_task=501574444,maximum_http_requests=119,
                      max_wall_clock_seconds=2316,max_decoded_csv_bytes=1166187496)
    parent['ancestor_completions']=bindings
    return parent


def test_entire_ancestor_chain_deducted_and_inventory_cannot_drop_grandparent(tmp_path,monkeypatch):
    plan=lineage_fixture(tmp_path,monkeypatch);bill=ancestor_bill(plan)
    assert bill['caps']==dict(max_download_bytes_new_task=501082594,maximum_http_requests=114,
                             max_wall_clock_seconds=2292,max_rows_to_index=6000000,max_decoded_csv_bytes=1043884804)
    plan['ancestor_completions'].pop()
    with pytest.raises(ValueError,match='inventory'):ancestor_bill(plan)


def test_ancestor_plan_tamper_and_missing_prior_bill_rejected(tmp_path,monkeypatch):
    plan=lineage_fixture(tmp_path,monkeypatch);bad=copy.deepcopy(plan);bad['prior_accounted_body_bound']-=1
    with pytest.raises(ValueError,match='accumulated bill'):ancestor_bill(bad)
    path=Path(plan['continuation_parent_completion']).parent/'plan.json';path.write_text('{}')
    with pytest.raises(ValueError,match='plan SHA'):ancestor_bill(plan)


def test_cyclic_parent_chain_rejected_before_recount(tmp_path,monkeypatch):
    from scripts import tgsim_bounded_transport as module
    plan=lineage_fixture(tmp_path,monkeypatch)
    parentpath=Path(plan['continuation_parent_completion']).parent/'plan.json'
    parentpath.write_text(json.dumps(plan))
    claimpath=parentpath.parent/'execution_claim.json';claim=json.loads(claimpath.read_text());claim['proposal']=plan
    claim['proposal_sha256']='fixed';claimpath.write_text(json.dumps(claim))
    compath=parentpath.parent/'completion.json';completion=json.loads(compath.read_text());completion['proposal_sha256']='fixed';compath.write_text(json.dumps(completion))
    plan['continuation_parent_completion_sha256']=plan['continuation_parent_plan_sha256']='fixed'
    parentpath.write_text(json.dumps(plan));claim['proposal']=plan;claimpath.write_text(json.dumps(claim))
    monkeypatch.setattr(module,'digest',lambda path:'fixed')
    with pytest.raises(ValueError,match='cyclic'):ancestor_bill(plan)


def test_full_acquisition_and_independent_attempt_consumer(tmp_path,monkeypatch):
    from tests.test_tgsim_paged_preparation import setup,FixtureTransport
    from scripts import audit_tgsim_paged_input as consumer
    from scripts.prepare_tgsim_paged_input import acquire,disk_bytes
    from scripts.recover_tgsim_paged_input import NativeComposer
    plan,raw,analysis,budget=setup(tmp_path,geometry=True)
    plan.update(automatic_retry=True,max_total_attempts_per_trajectory_page=3,max_extra_attempts_global=8,
                destination_raw_directory=str(raw),identity_mode='opaque_unique_sqlite',max_decoded_csv_bytes=100000)
    budget.maximum_requests=plan['maximum_http_requests']=29
    class Transport(FixtureTransport):
        def open(self,request,**kwargs):
            response=super().open(request,**kwargs)
            if '.csv?' in request.full_url and len(self.calls)==5:
                broken=Broken();broken.geturl=lambda:request.full_url;return broken
            return response
    fetcher=BoundedPageFetcher(plan,analysis);composer=NativeComposer(analysis,100000);opener=Transport(plan)
    result=acquire(plan,raw,analysis,budget,opener=opener,append_fn=composer,gzip_pages=True,page_fetcher=fetcher)
    composer.close();assert budget.rows==6 and budget.requests==29 and fetcher.extra_attempts==1
    proposal=tmp_path/'plan.json';save_new(proposal,plan);grant=tmp_path/'grant.json';save_new(grant,{})
    record=dict(proposal_sha256=digest(proposal),source_commit='fixture',grant_sha256=digest(grant),
                command_arguments={'proposal':str(proposal),'grant':str(grant)},training_frames=0,evaluation_frames=0,hidden_openings=0,ancestor_bill={})
    save_new(analysis/'execution_claim.json',dict(record,proposal=plan))
    completion=dict(record,**result,status='acquired_and_input_checked_not_science_qualified',http_requests=budget.requests,
                    extra_transport_attempts=1,rows_indexed=budget.rows,decoded_csv_bytes=composer.decoded,
                    total_body_download_bytes=budget.download_bytes,total_body_download_bytes_upper_bound=budget.download_bytes+budget.interrupted_body_upper_bound,
                    two_task_body_upper_bound=consumer.OLD_BODY_BOUND+budget.download_bytes+budget.interrupted_body_upper_bound,
                    cumulative_disk_bytes=disk_bytes([raw,analysis]),cumulative_disk_roots=[str(raw),str(analysis)],
                    raw_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in raw.iterdir()])
    save_new(analysis/'completion.json',completion)
    # Grant/ancestor validation is separately exercised above; isolate this consumer test.
    monkeypatch.setattr(consumer,'validate_retry_grant',lambda *args:{})
    audited=consumer.audit(analysis,expected_proposal_sha=digest(proposal))
    assert audited['acquisition_complete'] and audited['extra_transport_attempts']==1
    assert audited['network_http_requests']==29 and audited['native_indexed_rows_verified']==6
    stray=analysis/'attempts/stray.json';save_new(stray,{})
    with pytest.raises(ValueError,match='unaccounted'):consumer.audit(analysis,expected_proposal_sha=digest(proposal))


def test_uncertain_bound_limits_chunked_reads_without_content_length(tmp_path):
    _,_,_,budget,fetcher,path,receipt=fixture(tmp_path)
    budget.proposal['max_download_bytes']=10;budget.interrupted_body_upper_bound=8
    response=Response(b'abc');response.headers.pop('Content-Length')
    opener=Sequence(response,Response())
    with pytest.raises(BudgetLimit):fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    assert opener.calls==1 and budget.download_bytes==2 and budget.interrupted_body_upper_bound==8
    assert Path(str(path)+'.attempt_1').read_bytes()==b'ab'


def test_extra_attempt_limit_is_shared_between_pages(tmp_path):
    _,raw,analysis,budget,fetcher,path,receipt=fixture(tmp_path,max_extra_attempts_global=1)
    opener=Sequence(Broken(),Response(),Broken(),Response())
    fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    with pytest.raises(BudgetLimit,match='global extra'):
        fetcher(URL+'&offset=2',raw/'fixture_page_001.csv.body',budget,
                analysis/'fixture_page_001_receipt.json',opener=opener,read_timeout_seconds=30)
    assert opener.calls==3 and fetcher.extra_attempts==1


def test_consumer_rejects_semantic_error_retried_with_otherwise_valid_hashes(tmp_path):
    plan,raw,analysis,budget,fetcher,path,receipt=fixture(tmp_path)
    fetcher(URL,path,budget,receipt,opener=Sequence(Broken(),Response()),read_timeout_seconds=30)
    logical=json.loads(receipt.read_text());first=Path(logical['attempts'][0]['path'])
    attempt=json.loads(first.read_text());attempt['error_type']='ValueError';first.write_text(json.dumps(attempt))
    logical['attempts'][0]['sha256']=digest(first)
    with pytest.raises(ValueError,match='nontransport'):audit_attempts(logical,receipt,raw,analysis,plan)


def test_crc_failure_after_full_transfer_does_not_retry(tmp_path):
    import csv,gzip
    from scripts.recover_tgsim_paged_input import NativeComposer
    _,raw,analysis,budget,fetcher,path,receipt=fixture(tmp_path)
    response=Response(b'invalid gzip');response.headers['Content-Encoding']='gzip';opener=Sequence(response,Response())
    info=fetcher(URL,path,budget,receipt,opener=opener,read_timeout_seconds=30)
    composer=NativeComposer(analysis,10000)
    with pytest.raises(gzip.BadGzipFile):
        composer(path,{'asset_id':'fixture','expected_native_fields':['id','time']},csv.writer(io.StringIO()),1,None,budget,encoded_gzip=info['encoded_wire_body_retained'])
    composer.close();assert opener.calls==1 and fetcher.extra_attempts==0


def test_generic_or_denied_grant_cannot_activate_bounded_transport(tmp_path):
    from scripts.tgsim_bounded_transport import validate_retry_grant
    with pytest.raises((KeyError,ValueError)):
        validate_retry_grant({}, {'proposal':str(tmp_path/'missing'),'execution_authorized':True})
