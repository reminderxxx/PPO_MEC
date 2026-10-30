"""Eight new offline cases for the explicit one-request extension, no old reruns."""
import io
import json
from pathlib import Path

import pytest

from scripts import acquire_public_task_metadata as producer
from scripts import audit_public_task_metadata as consumer
from scripts import run_public_task_metadata_extension as runner

B = Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_evidence_strengthening_20261029_v1/quality_public_sample_acquisition_20261030_v1')


class Response(io.BytesIO):
    def __init__(self, body=b'[]', status=200, mime='application/json', disconnect=False):
        super().__init__(body); self.status=status; self.headers={'Content-Type':mime}; self.disconnect=disconnect
    def geturl(self):return producer.PUBLIC_FILES
    def read1(self, size):
        if self.disconnect:raise ConnectionError('synthetic interruption')
        return super().read1(size)


@pytest.fixture
def case(tmp_path, monkeypatch):
    plan=json.loads((B/'metadata_plan_v3.json').read_text())
    plan.update(raw_root=str(tmp_path/'fresh'), audit_output=str(tmp_path/'audit.json'),
                completion_output=str(tmp_path/'completion.json'), storage_roots=plan['storage_roots']+[str(tmp_path)],
                extension_boot_session_uuid=producer.extension_boot_id(),
                metadata_caps={'http_requests':8,'body_bytes':1980613,'wall_seconds':120},
                single_route_extension={'proposal':str(B/'single_route_extension_proposal.json'),
                                        'proposal_sha256':producer.EXTENSION_SHA,
                                        'predecessor_plan':str(B/'metadata_plan_v3.json')})
    pp=tmp_path/'plan.json'; gp=tmp_path/'grant.json'; producer.save(pp,plan)
    grant={'plan_sha256':producer.digest(pp),'proposal_sha256':producer.PROPOSAL_SHA,
           'allow_metadata':True,'allow_image_downloads':False,
           'extension_proposal_sha256':producer.EXTENSION_SHA,'allow_single_route_extension':True}
    producer.save(gp,grant); calls=[]
    monkeypatch.setattr(producer,'source_check',lambda p:None)
    class Opener:
        response=Response()
        def open(self,*args,**kwargs):calls.append(args[0].full_url);return self.response
    op=Opener(); monkeypatch.setattr(producer.urllib.request,'build_opener',lambda *a:op)
    return plan,pp,gp,op,calls


def test_extension_one_success_and_automatic_consumer(case):
    p,pp,gp,_,calls=case;r=runner.run(pp,producer.digest(pp),gp);a=json.loads(Path(p['audit_output']).read_text())
    assert r['producer_success'] and r['audit_success'] and r['new_deadline_pass']
    assert calls==[producer.PUBLIC_FILES] and a['new_requests']==1 and a['requests']==8 and a['body_bytes']==932039
    assert a['old_metadata_deadline_pass'] is False and a['old_metadata_deadline_elapsed_seconds']>600 and a['D_wall_remaining'] is None
    assert a['run_start_clock']==json.loads((Path(p['raw_root'])/'claim.json').read_text())['extension_start_clock']
    with pytest.raises(ValueError,match='single launch'):runner.run(pp,producer.digest(pp),gp)


def test_extension_explicit_grant_before_http(case):
    p,pp,gp,_,calls=case;g=json.loads(gp.read_text());g['allow_single_route_extension']=False;gp.write_text(json.dumps(g))
    result=runner.run(pp,producer.digest(pp),gp)
    assert not calls and not result['producer_success'] and not result['audit_success'] and result['new_request_upper_on_audit_failure']==0


def test_extension_no_recursive_or_wrong_parent(case):
    p,pp,_,_,calls=case;p['single_route_extension']['predecessor_plan']=str(B/'metadata_plan_v2.json')
    with pytest.raises((ValueError,KeyError)):producer.parent_ledger(p)
    assert not calls


def test_extension_schema_caps_and_url_are_exact(case):
    p,_,_,_,calls=case;p['metadata_caps']['http_requests']=9
    with pytest.raises(ValueError,match='exact extension'):producer.validate_schema(p)
    p['metadata_caps']['http_requests']=8
    with pytest.raises(ValueError,match='only the one'):producer.validate_intent(Path(p['raw_root']),p,{'url':next(iter(producer.SEEDS))},[])
    assert not calls


def test_extension_disconnect_charged_and_stopped(case):
    p,pp,gp,op,calls=case;op.response=Response(disconnect=True);result=runner.run(pp,producer.digest(pp),gp);a=json.loads(Path(p['audit_output']).read_text())
    assert len(calls)==1 and not result['producer_success'] and result['audit_success']
    assert a['requests']==8 and a['body_bytes']==932037 and a['interrupted_read_upper_bound']==65536 and a['request_failures']==1


def test_extension_binary_error_never_consumed(case):
    p,pp,gp,op,calls=case;op.response=Response(b'fake binary',400,'image/jpeg');r=runner.run(pp,producer.digest(pp),gp);a=json.loads(Path(p['audit_output']).read_text())
    assert len(calls)==1 and not r['producer_success'] and r['audit_success'] and a['body_bytes']==932037 and a['interrupted_read_upper_bound']==65536
    assert (Path(p['raw_root'])/'request_000/body.bin').stat().st_size==0


def test_extension_expired_claim_cannot_restart_at_get(case,monkeypatch):
    p,pp,gp,_,calls=case;root=Path(p['raw_root']);producer.initialize(pp,producer.digest(pp),gp,root)
    claim=json.loads((root/'claim.json').read_text());claim['extension_start_clock']-=121;(root/'claim.json').write_text(json.dumps(claim))
    intent=root/'intent.json';producer.save(intent,{'url':producer.PUBLIC_FILES,'source':'edgevision_v1','kind':'metadata'})
    with pytest.raises(TimeoutError):producer.acquire_one(root,intent,producer.digest(intent),verify_source=False)
    assert not calls and not list(root.glob('request_*'))
    a=consumer.audit(root,producer.digest(pp),'expired claim');assert a['within_metadata_deadline'] is False and a['new_requests']==0 and a['requests']==7


def test_extension_one_mib_includes_uncertain_read(case):
    p,pp,gp,op,calls=case;op.response=Response();op.response.headers['Content-Length']='1048576'
    r=runner.run(pp,producer.digest(pp),gp);a=json.loads(Path(p['audit_output']).read_text())
    assert len(calls)==1 and not r['producer_success'] and r['audit_success'] and a['body_bytes']+a['interrupted_read_upper_bound']<=1980613


def test_extension_closure_uuid_success_and_shared_clock(case,monkeypatch):
    p,pp,gp,_,calls=case
    monkeypatch.setattr(producer,'boot_id',lambda:'historical boottime usec differs; never compare to live UUID')
    r=runner.run(pp,producer.digest(pp),gp);a=json.loads(Path(p['audit_output']).read_text());claim=json.loads((Path(p['raw_root'])/'claim.json').read_text())
    assert r['producer_success'] and r['audit_success'] and len(calls)==1
    assert a['requests']==8 and a['body_bytes']==932039 and a['old_metadata_deadline_elapsed_seconds']>600 and a['old_metadata_deadline_pass'] is False
    assert a['D_wall_remaining'] is None and a['run_start_clock']==claim['extension_start_clock'] and claim['boot_id_kind']=='kern.bootsessionuuid'
    with pytest.raises(ValueError,match='single launch'):runner.run(pp,producer.digest(pp),gp)


def test_extension_closure_terminal_commit_failure_is_not_audit_success(case,monkeypatch):
    p,pp,gp,_,calls=case;original=producer.save
    def save(path,value):
        if path.name=='terminal.json':raise OSError('synthetic terminal write failure')
        return original(path,value)
    monkeypatch.setattr(producer,'save',save)
    r=runner.run(pp,producer.digest(pp),gp)
    assert len(calls)==1 and not r['audit_success'] and r['audit'] is None and Path(p['audit_output']).exists()
    assert r['body_upper_on_audit_failure']==1048576 and r['new_request_upper_on_audit_failure']==1


def test_extension_closure_partial_claim_keeps_completion(case,monkeypatch):
    p,pp,gp,_,calls=case;original=producer.save
    def save(path,value):
        if path.name=='claim.json':
            path.write_text('{');raise OSError('synthetic partial claim failure')
        return original(path,value)
    monkeypatch.setattr(producer,'save',save)
    r=runner.run(pp,producer.digest(pp),gp)
    assert not calls and not r['producer_success'] and not r['audit_success'] and Path(p['completion_output']).exists()
    assert r['new_request_upper_on_audit_failure']==0 and any(x['stage']=='claim' for x in r['errors'])
