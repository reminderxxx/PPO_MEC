"""Synthetic frozen diagnostic sequence, never an external capability query."""
import csv
import hashlib
import io
import json
import urllib.error

import pytest

from scripts import diagnose_tgsim_query as diagnostic


@pytest.mark.parametrize('message,success,expected_calls',[
    ('Parse error: unexpected wildcard',True,3),('Permission denied',False,1)])
def test_frozen_three_request_sequence_or_single_unknown_rejection(tmp_path,monkeypatch,message,success,expected_calls):
    class Response(io.BytesIO):
        status=200
        def __init__(self,body,url):
            super().__init__(body);self.url=url;self.headers={'Content-Type':'text/csv','Content-Length':str(len(body))}
        def geturl(self):return self.url
    class Opener:
        calls=0
        def open(self,request,**kwargs):
            self.calls+=1
            if self.calls==1:
                body=json.dumps({'message':message}).encode()
                raise urllib.error.HTTPError(request.full_url,400,'Bad Request',{'Content-Type':'application/json','Content-Length':str(len(body))},io.BytesIO(body))
            return Response(b'source_row_id,id,time\nrow-abc,1,0.0\n',request.full_url)
    opener=Opener();monkeypatch.setattr(diagnostic.urllib.request,'build_opener',lambda *args:opener)
    monkeypatch.setattr(diagnostic,'ROOT',tmp_path)
    monkeypatch.setattr(diagnostic.subprocess,'check_output',lambda args,**kwargs:'' if 'status' in args else 'synthetic-commit')
    plan=dict(output=str(tmp_path/'new_output'),maximum_requests=3,max_download_bytes=262144,max_wall_clock_seconds=120,
              max_new_disk_bytes=2147483648,max_trajectory_rows_to_index=0,disk_roots=[str(tmp_path)],training_frames=0,evaluation_frames=0,
              requests=[dict(url=f'https://data.transportation.gov/resource/fixture.csv?i={i}',expected_native_fields=['id','time']) for i in range(3)])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if success:diagnostic.execute(path,sha)
    else:
        with pytest.raises(ValueError,match='not an identified syntax'):diagnostic.execute(path,sha)
    result=json.loads((tmp_path/'new_output/completion.json').read_text())
    assert opener.calls==expected_calls
    assert (result['status']=='syntax_diagnosed_and_two_explicit_native_probes_pass')==success
    assert result['training_frames']==result['evaluation_frames']==0
    assert json.loads((tmp_path/'new_output/observation_0.body').read_text())['message']==message
