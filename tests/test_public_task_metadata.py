"""Bounded synthetic transport/provenance tests; no public network/model calls."""
import io
import json
import urllib.error

import pytest

from scripts import acquire_public_task_metadata as producer
from scripts.audit_public_task_metadata import audit


class Response(io.BytesIO):
    def __init__(self, data=b'{}', status=200, mime='application/json', length=None, error=False):
        super().__init__(data); self.status = status; self.error = error
        self.headers = {'Content-Type': mime}
        if length is not None:self.headers['Content-Length'] = str(length)
    def geturl(self):return 'https://data.mendeley.com/datasets/j82bnw7gsr/1'
    def read1(self, size):
        if self.error:raise ConnectionError('synthetic disconnect')
        return super().read1(size)


class Opener:
    def __init__(self, response):self.response=response;self.calls=0
    def open(self, *args, **kwargs):self.calls+=1;return self.response


@pytest.fixture
def case(tmp_path):
    root=tmp_path/'run';root.mkdir()
    plan={'metadata_caps':{'http_requests':12,'body_bytes':5242880,'wall_seconds':600},
          'proposal_sha256':producer.PROPOSAL_SHA,'storage_cap':268435456,'allowed_hosts':sorted(producer.HOSTS),
          'seed_urls':sorted(producer.SEEDS),'storage_roots':[str(tmp_path)],'worktree':str(tmp_path),
          'source_commit':'a'*40,'source_sha256':{'scripts/acquire_public_task_metadata.py':'0'*64,'scripts/audit_public_task_metadata.py':'0'*64}}
    producer.save(root/'plan.json',plan);h=producer.digest(root/'plan.json')
    producer.save(root/'grant.json',{'plan_sha256':h,'proposal_sha256':producer.PROPOSAL_SHA,'allow_metadata':True,'allow_image_downloads':False})
    producer.save(root/'claim.json',{'plan_sha256':h,'grant_sha256':producer.digest(root/'grant.json'),'boot_id':producer.boot_id()})
    intent=tmp_path/'intent.json';producer.save(intent,{'url':'https://data.mendeley.com/datasets/j82bnw7gsr/1','source':'edgevision_v1','kind':'metadata'})
    def run(response=None):
        op=Opener(response or Response());producer.acquire_one(root,intent,producer.digest(intent),opener=op,verify_source=False);return op
    return root,h,intent,run


def test_complete_and_independent_body_check(case):
    root,h,_,run=case;op=run();result=audit(root,h,'inspect saved metadata')
    assert op.calls==1 and result['requests']==1 and result['body_bytes']==2
    assert not result['input_qualified'] and result['D_request_remaining']==31


@pytest.mark.parametrize('response',[
    Response(b'bad request',status=400),Response(b'<redirect>',status=302),
    Response(b'{}',mime='image/jpeg'),Response(b'{}',length=10)])
def test_failure_retained_and_never_retried(case,response):
    root,h,_,run=case
    with pytest.raises(ValueError):run(response)
    r=json.loads((root/'request_000/receipt.json').read_text());assert r['status']=='failed'
    assert audit(root,h,'failure')['request_failures']==1
    with pytest.raises(ValueError,match='failed'):run()
    assert len(list(root.glob('request_*')))==1


def test_disconnect_charges_uncertain_read(case):
    root,h,_,run=case
    with pytest.raises(ConnectionError):run(Response(error=True))
    r=audit(root,h,'failure');assert r['interrupted_read_upper_bound']>0


def test_body_tamper_rejected(case):
    root,h,_,run=case;run();(root/'request_000/body.bin').write_bytes(b'xx')
    with pytest.raises(AssertionError,match='hash'):audit(root,h,'tamper')


def test_duplicate_url_rejected_before_network(case):
    root,_,_,run=case;run()
    with pytest.raises(ValueError,match='duplicate'):run()
    assert len(list(root.glob('request_*')))==1


def test_source_link_must_be_literal(case):
    root,_,intent,_=case
    intent.write_text(json.dumps({'url':'https://data.mendeley.com/public-api/datasets/j82bnw7gsr/versions/1/files','source':'edgevision_v1','kind':'metadata','proof':{'request_index':0,'body_sha256':'x'}}))
    with pytest.raises(ValueError,match='parent'):producer.acquire_one(root,intent,producer.digest(intent),verify_source=False)
    assert not list(root.glob('request_*'))


def test_deadline_cannot_reset(case):
    root,_,_,run=case;run();p=root/'request_000/receipt.json';r=json.loads(p.read_text())
    r['run_start_clock']=producer.clock()-601;p.write_text(json.dumps(r))
    # Another fixed seed avoids URL de-duplication masking the time guard.
    intent=root.parent/'next.json';intent.write_text(json.dumps({'url':'https://data.mendeley.com/api/docs/','source':'edgevision_v1','kind':'documentation'}))
    with pytest.raises(TimeoutError):producer.acquire_one(root,intent,producer.digest(intent),verify_source=False)
    assert len(list(root.glob('request_*')))==1


def test_declared_overflow_and_encoding(case):
    root,h,_,run=case
    with pytest.raises(ValueError,match='reserve'):run(Response(length=5242880))
    assert audit(root,h,'overflow')['interrupted_read_upper_bound']==65536


def test_forged_success_mime_rejected(case):
    root,h,_,run=case;run();p=root/'request_000/receipt.json';r=json.loads(p.read_text());r['headers']['Content-Type']='image/png';p.write_text(json.dumps(r))
    with pytest.raises(AssertionError,match='MIME'):audit(root,h,'tamper')


def test_terminal_and_plan_immutability(case):
    root,_,intent,run=case;producer.save(root/'terminal.json',{})
    with pytest.raises(ValueError,match='terminal'):run()
    assert not list(root.glob('request_*'))


def test_authorization_binding(case):
    root,h,_,run=case;run();grant=json.loads((root/'grant.json').read_text());grant['proposal_sha256']='wrong';(root/'grant.json').write_text(json.dumps(grant))
    claim=json.loads((root/'claim.json').read_text());claim['grant_sha256']=producer.digest(root/'grant.json');(root/'claim.json').write_text(json.dumps(claim))
    with pytest.raises(AssertionError,match='authorization'):audit(root,h,'tamper')


def test_closure_caps_before_network(case):
    root,_,intent,_=case;plan=json.loads((root/'plan.json').read_text());plan['metadata_caps']['body_bytes']=5242881
    (root/'plan.json').write_text(json.dumps(plan));op=Opener(Response())
    with pytest.raises(ValueError,match='cap'):producer.acquire_one(root,intent,producer.digest(intent),opener=op,verify_source=False)
    assert op.calls==0 and not list(root.glob('request_*'))


def test_closure_grant_before_network(case):
    root,_,intent,_=case;grant=json.loads((root/'grant.json').read_text());grant['allow_metadata']=False
    (root/'grant.json').write_text(json.dumps(grant));op=Opener(Response())
    with pytest.raises(ValueError,match='binding'):producer.acquire_one(root,intent,producer.digest(intent),opener=op,verify_source=False)
    assert op.calls==0 and not list(root.glob('request_*'))


def test_closure_binary_error_and_redirect_policy(case):
    root,h,intent,_=case
    class ErrorOpener:
        calls=0
        def open(self,*args,**kwargs):
            self.calls+=1
            raise urllib.error.HTTPError('https://data.mendeley.com/datasets/j82bnw7gsr/1',400,'bad',{'Content-Type':'image/jpeg'},io.BytesIO(b'\xff\xd8binary'))
    op=ErrorOpener()
    with pytest.raises(ValueError,match='MIME'):producer.acquire_one(root,intent,producer.digest(intent),opener=op,verify_source=False)
    result=audit(root,h,'binary error');assert op.calls==1 and result['body_bytes']==0 and result['interrupted_read_upper_bound']==65536
    assert producer.NoRedirect().redirect_request(None,None,302,'',{},'https://other.invalid') is None


def test_closure_other_dataset_even_if_literal(case):
    root,_,intent,run=case;other='https://data.mendeley.com/public-api/datasets/OTHER/versions/1/files';run(Response(other.encode()))
    intent.write_text(json.dumps({'url':other,'source':'edgevision_v1','kind':'metadata','proof':{'request_index':0,'body_sha256':producer.digest(root/'request_000/body.bin')}}));op=Opener(Response())
    with pytest.raises(ValueError,match='dataset'):producer.acquire_one(root,intent,producer.digest(intent),opener=op,verify_source=False)
    assert op.calls==0 and len(list(root.glob('request_*')))==1


def test_closure_clock_and_external_bill(case,monkeypatch):
    root,h,intent,run=case;run();rp=root/'request_000/receipt.json';r=json.loads(rp.read_text());original=r.copy()
    r['finished_clock']=producer.clock()+30;rp.write_text(json.dumps(r))
    intent.write_text(json.dumps({'url':'https://data.mendeley.com/api/docs/','source':'edgevision_v1','kind':'documentation'}))
    with pytest.raises(ValueError,match='clock'):producer.acquire_one(root,intent,producer.digest(intent),verify_source=False)
    with pytest.raises(AssertionError,match='clock'):audit(root,h,'clock')
    rp.write_text(json.dumps(original));(root.parent/'external_control.log').write_bytes(b'x'*10000)
    result=audit(root,h,'normal');assert result['retained_bytes_before_audit']>=10000
    assert result['D_storage_remaining_conservative']==268435456-result['retained_bytes_before_audit']-1048576
    monkeypatch.setattr(producer.time,'clock_gettime',lambda unused:original['run_start_clock']+601)
    late=audit(root,h,'late');assert late['status']=='FAILED_METADATA_DEADLINE_LEDGER_RETAINED' and late['D_wall_remaining']==1199


def test_closure_init_exact_bytes_and_boot(case,monkeypatch):
    root,h,_,_=case;plan=root/'plan.json';grant=root/'grant.json';calls=[]
    monkeypatch.setattr(producer,'source_check',lambda x:calls.append(x['source_commit']))
    dest=root.parent/'initialized';producer.initialize(plan,h,grant,dest)
    assert calls==['a'*40] and dest.joinpath('plan.json').read_bytes()==plan.read_bytes()
    producer.check_binding(dest,json.loads(plan.read_text()))
    monkeypatch.setattr(producer,'boot_id',lambda:'DIFFERENT_BOOT')
    with pytest.raises(ValueError,match='boot'):producer.check_binding(dest,json.loads(plan.read_text()))
