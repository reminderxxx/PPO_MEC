"""Three offline cases for publisher-backed endpoint diagnosis, no public requests."""
import io
import json
from pathlib import Path

import pytest

from scripts import diagnose_public_task_entry as entry

B=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_evidence_strengthening_20261029_v1/quality_public_entry_diagnosis_20261030_v1')


class Response(io.BytesIO):
    def __init__(self,index,body,length=None):
        super().__init__(body);self.url=(entry.BUNDLE,entry.FILES)[index];self.status=200
        self.headers={'Content-Type':'application/javascript' if index==0 else 'application/json'}
        if length is not None:self.headers['Content-Length']=str(length)
    def geturl(self):return self.url


class Opener:
    def __init__(self,response):self.response=response;self.calls=0
    def open(self,*args,**kwargs):self.calls+=1;return self.response


@pytest.fixture
def case(tmp_path):
    page=tmp_path/'page.txt';page.write_text(entry.BUNDLE.split('mendeley.com')[1])
    p={'raw_root':str(tmp_path/'run'),'contract':str(B/'diagnosis_contract.json'),'contract_sha256':entry.common.digest(B/'diagnosis_contract.json'),
       'official_page':str(page),'official_page_sha256':entry.common.digest(page),'storage_roots':[str(tmp_path)]}
    pp=tmp_path/'plan.json';entry.common.save(pp,p);entry.init(pp,entry.common.digest(pp),verify_source=False)
    root=Path(p['raw_root'])
    def bundle():
        text='function get(){return {folder_id:currentFolder,root:"root",url:"/files"};}'
        op=Opener(Response(0,text.encode()));entry.get(root,0,opener=op,verify_source=False)
        h=entry.common.digest(root/'request_0/body.bin');proof=tmp_path/'proof.json';entry.common.save(proof,{'bundle_sha256':h,'corrected_url':entry.FILES,'snippets':[{'start':0,'end':len(text),'text':text}],
             'independent_semantic_review':{'reviewer':'/root/quality_source_independent_check','bundle_sha256':h,'public_files_call_verified':True,'root_folder_relationship_verified':True,'decision':'ACCEPTED_PUBLISHER_CALL_RELATIONSHIP','scope':'synthetic software example only'}})
        return proof
    return root,tmp_path,bundle


def test_diagnosis_two_requests_and_cumulative_bill(case):
    root,tmp,bundle=case;proof=bundle();op=Opener(Response(1,b'[]'));entry.get(root,1,proof,entry.common.digest(proof),opener=op,verify_source=False)
    a=entry.audit(root,tmp/'audit.json');assert op.calls==1 and a['cumulative_requests']==10 and a['new_requests']==2 and a['images']==0
    assert a['cumulative_body_bytes']==932050+a['new_body_bytes'] and not a['input_qualified']
    with pytest.raises(AssertionError,match='terminal'):entry.get(root,1,proof,opener=op,verify_source=False)


def test_diagnosis_api_error_is_retained_and_stops(case):
    root,tmp,bundle=case;proof=bundle();op=Opener(Response(1,b'{"error":400}'))
    with pytest.raises(AssertionError,match='API error'):entry.get(root,1,proof,entry.common.digest(proof),opener=op,verify_source=False)
    a=entry.audit(root,tmp/'audit.json');assert a['failures']==1 and op.calls==1 and a['cumulative_requests']==10
    assert (root/'request_1/body.bin').read_text()=='{"error":400}'


def test_diagnosis_declared_cap_rejected_before_body_read(case):
    root,tmp,_=case;op=Opener(Response(0,b'not consumed',length=3145728))
    with pytest.raises(AssertionError,match='declared response cap'):entry.get(root,0,opener=op,verify_source=False)
    a=entry.audit(root,tmp/'audit.json');assert a['new_body_bytes']==0 and a['new_interrupted_upper']==65536 and a['cumulative_requests']==9
