"""Software-only budget/provenance/time checks. No network or project environment."""
import csv
import http.client
import io
import json
import urllib.error
from pathlib import Path

import pytest

from scripts.prepare_tgsim_input import (Budget, BudgetLimit, fetch, feasible_starts,
    inspect_trajectory, metadata_pin, temporal_packing, validate_plan, validate_url)


def budget(tmp_path, **overrides):
    limits = dict(max_wall_clock_seconds=30, max_new_disk_bytes=32*1024**2,
                  max_download_bytes=10000, max_trajectory_rows_to_index=20000)
    limits.update(overrides)
    return Budget(limits, [tmp_path])


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, *, length=None):
        super().__init__(data)
        self.headers = {'Content-Type': 'text/csv'}
        if length is not None:self.headers['Content-Length'] = str(length)

    def geturl(self):return 'https://data.transportation.gov/api/assets/example'


class Opener:
    def __init__(self, response):self.response = response; self.calls = 0
    def open(self, *args, **kwargs):self.calls += 1; return self.response


def trajectory(tmp_path, *, count=480, vehicles=8, gap=None, duplicate=False, bad_value=None):
    names = ['id','time','xloc_kf','yloc_kf','speed_kf','run_index']
    source = {'asset_id':'fixture', 'metadata_pin': {'column_specs':[
        {'name':name,'dataTypeName':'number'} for name in names]}}
    path = tmp_path/'fixture.csv'
    with path.open('w', newline='') as out:
        writer = csv.writer(out);writer.writerow(names)
        for v in range(vehicles):
            for tick in range(count):
                if gap == (v,tick):continue
                row = [v, str(tick//10)+'.'+str(tick%10), bad_value if bad_value and tick==2 else v, tick, 10, 1]
                writer.writerow(row)
                if duplicate and v==0 and tick==0:writer.writerow(row)
    return path, source


def test_complete_native_ticks_and_persistent_fleet(tmp_path):
    path,source = trajectory(tmp_path)
    r = inspect_trajectory(path,source,tmp_path/'index.sqlite',budget(tmp_path))
    assert r['rows']==3840
    assert r['runs'][1]['native_480_tick_start_ranges']==[[0,0]]
    assert r['runs'][1]['vehicles']==8
    assert r['runs'][1]['certified_formal_or_hidden']==0
    assert r['endpoint_span_seconds']=='47.9'
    assert r['geometry_gate'].startswith('UNQUALIFIED')


def test_one_missing_tick_is_not_filled(tmp_path):
    path,source = trajectory(tmp_path,gap=(0,240))
    r = inspect_trajectory(path,source,tmp_path/'index.sqlite',budget(tmp_path))
    assert r['runs'][1]['native_480_tick_start_ranges']==[]
    assert r['runs'][1]['missing_vehicle_ticks']==1


def test_run_relative_clocks_and_vehicle_ids_stay_separate(tmp_path):
    path,source = trajectory(tmp_path,vehicles=4)
    lines=path.read_text().splitlines()
    with path.open('a') as out:
        for line in lines[1:]:out.write(line.rsplit(',',1)[0]+',2\n')
    r=inspect_trajectory(path,source,tmp_path/'index.sqlite',budget(tmp_path))
    assert set(r['runs'])=={1,2}
    assert all(x['vehicles']==4 and not x['native_480_tick_start_ranges'] for x in r['runs'].values())


def test_duplicates_stop_without_repair(tmp_path):
    path,source = trajectory(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='duplicate'):
        inspect_trajectory(path,source,tmp_path/'index.sqlite',budget(tmp_path))
    assert path.exists()


@pytest.mark.parametrize('value',['nan','inf','-inf'])
def test_nonfinite_rejected(tmp_path,value):
    path,source = trajectory(tmp_path,count=3,vehicles=1,bad_value=value)
    with pytest.raises(ValueError,match='nonfinite'):
        inspect_trajectory(path,source,tmp_path/'index.sqlite',budget(tmp_path))


def test_fractional_time_not_truncated(tmp_path):
    path,source = trajectory(tmp_path,count=3,vehicles=1)
    path.write_text(path.read_text().replace('0.1,','0.15,'))
    with pytest.raises(ValueError,match='off-grid'):
        inspect_trajectory(path,source,tmp_path/'index.sqlite',budget(tmp_path))


def test_row_cap_aborts_task_not_source_only(tmp_path):
    path,source = trajectory(tmp_path,count=3,vehicles=1)
    b = budget(tmp_path,max_trajectory_rows_to_index=2)
    with pytest.raises(BudgetLimit,match='row cap'):
        inspect_trajectory(path,source,tmp_path/'index.sqlite',b)
    assert b.rows==2


def test_source_runs_do_not_merge_and_gap_packing_is_explicit():
    assert feasible_starts([(0,479)]*7)==[]
    assert feasible_starts([(0,1439)]*8)==[[0,960]]
    assert temporal_packing([[0,960]])==2
    assert feasible_starts([(0,479)]*8)==[[0,0]]


def test_download_failure_preserves_body_receipt_and_never_retries(tmp_path):
    opener=Opener(Response(b'123',length=5));path=tmp_path/'raw.csv';receipt=tmp_path/'receipt.json'
    with pytest.raises(ValueError,match='truncated'):
        fetch('https://data.transportation.gov/api/assets/example',path,budget(tmp_path),receipt,opener=opener)
    assert opener.calls==1 and path.read_bytes()==b'123'
    assert json.loads(receipt.read_text())['status']=='failed'


def test_byte_cap_does_not_read_oversized_body(tmp_path):
    opener=Opener(Response(b'123456',length=6));b=budget(tmp_path,max_download_bytes=5)
    with pytest.raises(BudgetLimit,match='remaining byte budget'):
        fetch('https://data.transportation.gov/api/assets/example',tmp_path/'raw.csv',b,tmp_path/'receipt.json',opener=opener)
    assert opener.calls==1 and b.download_bytes==0


def test_create_only_and_host_guard(tmp_path):
    path=tmp_path/'raw.csv';path.write_text('preserved')
    opener=Opener(Response(b'new'))
    with pytest.raises(FileExistsError):
        fetch('https://data.transportation.gov/api/assets/example',path,budget(tmp_path),tmp_path/'receipt.json',opener=opener)
    assert path.read_text()=='preserved'
    assert opener.calls==0
    for url in ['http://data.transportation.gov/a','https://evil.example/a','https://data.transportation.gov@evil.example/a']:
        with pytest.raises(ValueError):validate_url(url)


def test_approval_is_bound_to_exact_proposal(tmp_path):
    path=tmp_path/'proposal.json';path.write_text('{}')
    with pytest.raises(ValueError,match='exact human-approved SHA'):
        validate_plan(path,tmp_path/'grant.json')


def test_metadata_pin_ignores_counters_but_binds_version_and_attachments():
    meta=dict(id='example',rowsUpdatedAt=1,downloadCount=3,columns=[],metadata={'attachments':[{'assetId':'a','name':'geometry.csv'}]})
    pin=metadata_pin(meta);meta['downloadCount']=9;assert metadata_pin(meta)==pin
    meta['rowsUpdatedAt']=2;assert metadata_pin(meta)!=pin


def test_total_wall_clock_and_disk_caps_are_hard(tmp_path):
    now=[0.]
    b=budget(tmp_path)
    b.clock=lambda:now[0];b.started=0.
    now[0]=30.
    with pytest.raises(TimeoutError):b.guard()
    now[0]=0.
    b.proposal['max_new_disk_bytes']=8*1024**2
    (tmp_path/'new_byte').write_bytes(b'x')
    with pytest.raises(BudgetLimit,match='disk'):b.guard()


def test_geometry_names_are_not_identity_keys():
    meta=dict(id='example',rowsUpdatedAt=1,columns=[],metadata={'attachments':[
        {'assetId':'a','name':'same.csv'},{'assetId':'b','name':'same.csv'}]})
    pin=metadata_pin(meta)
    assert len(pin['support_attachment_identities'])==2
    assert {x['assetId'] for x in pin['support_attachment_identities']}=={'a','b'}


class NativeChunkedResponse:
    """Exercise the standard-library parser that produced the observed failure."""
    def __init__(self, body):
        class Socket:
            def makefile(self, *args, **kwargs):
                return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nTransfer-Encoding: chunked\r\n\r\n'+body)
        self.response=http.client.HTTPResponse(Socket());self.response.begin()
        self.headers=self.response.headers;self.status=self.response.status
    def __enter__(self):return self
    def __exit__(self,*args):self.response.close()
    def geturl(self):return 'https://data.transportation.gov/api/assets/example'
    def read1(self,size):return self.response.read1(size)


def test_native_chunk_truncation_retains_payload_and_bounds_uncertainty(tmp_path):
    opener=Opener(NativeChunkedResponse(b'8\r\nabc'))
    b=budget(tmp_path,max_download_bytes=10);raw=tmp_path/'raw.csv';receipt=tmp_path/'receipt.json'
    with pytest.raises(http.client.IncompleteRead):
        fetch('https://data.transportation.gov/api/assets/example',raw,b,receipt,opener=opener)
    saved=json.loads(receipt.read_text())
    assert opener.calls==1 and raw.read_bytes()==b'abc'
    assert saved['bytes']==b.download_bytes==3
    assert saved['cumulative_body_bytes_upper_bound']==10
    assert saved['interrupted_read_body_upper_bound']==7
    assert b.interrupted_body_upper_bound==7
    assert saved['status']=='failed'


def test_native_complete_chunked_response_has_no_unresolved_payload(tmp_path):
    opener=Opener(NativeChunkedResponse(b'3\r\nabc\r\n0\r\n\r\n'))
    saved=fetch('https://data.transportation.gov/api/assets/example',tmp_path/'raw.csv',budget(tmp_path),tmp_path/'receipt.json',opener=opener)
    assert saved['status']=='complete' and saved['bytes']==3
    assert saved['interrupted_read_body_upper_bound']==0
    assert saved['cumulative_body_bytes_upper_bound']==3


def test_http_error_entity_and_headers_are_retained_from_same_request(tmp_path):
    url='https://data.transportation.gov/api/assets/example';body=b'{"message":"parse error"}'
    error=urllib.error.HTTPError(url,400,'Bad Request',{'Content-Type':'application/json','Content-Length':str(len(body))},io.BytesIO(body))
    class FailedOpener:
        calls=0
        def open(self,*args,**kwargs):self.calls+=1;raise error
    opener=FailedOpener();b=budget(tmp_path);raw=tmp_path/'error_body';receipt=tmp_path/'receipt.json'
    with pytest.raises(ValueError,match='HTTP400'):
        fetch(url,raw,b,receipt,opener=opener)
    saved=json.loads(receipt.read_text())
    assert opener.calls==1 and raw.read_bytes()==body and b.download_bytes==len(body)
    assert saved['status_code']==400 and saved['status']=='failed'
    assert saved['diagnostic_body_only'] and saved['diagnostic_entity_complete']
    assert saved['response_headers']['Content-Type']=='application/json'


def test_large_http_error_is_bounded_and_never_successful_input(tmp_path):
    url='https://data.transportation.gov/api/assets/example';body=b'x'*70000
    response=Response(body,length=len(body));response.status=400;opener=Opener(response)
    b=budget(tmp_path,max_download_bytes=80000);raw=tmp_path/'diagnostic';receipt=tmp_path/'receipt.json'
    with pytest.raises(ValueError,match='HTTP400'):
        fetch(url,raw,b,receipt,opener=opener)
    saved=json.loads(receipt.read_text())
    assert opener.calls==1 and raw.stat().st_size==65536
    assert saved['status']=='failed' and not saved['diagnostic_entity_complete']
    assert saved['bytes']==b.download_bytes==65536
