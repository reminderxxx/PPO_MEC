"""Native opaque IDs, encoded-wire accounting and bounded decompression."""
import csv
import gzip
import io
import json

import pytest

from scripts.prepare_tgsim_input import Budget, BudgetLimit, fetch
from scripts.prepare_tgsim_paged_input import system_id_key
from scripts.recover_tgsim_paged_input import NativeComposer,effective_caps


def budget(tmp_path):
    return Budget(dict(max_wall_clock_seconds=30,max_new_disk_bytes=32*1024**2,
                       max_download_bytes=100000,max_trajectory_rows_to_index=1000),[tmp_path])


def test_opaque_official_id_separators_and_nonlexical_page_order(tmp_path):
    composer=NativeComposer(tmp_path,10000);out=io.StringIO();writer=csv.writer(out)
    source=dict(asset_id='fixture',expected_native_fields=['id','time'])
    assert system_id_key('row-iwer.8rrb_8kf5')
    for index,sid in enumerate(['row-z.x_y','row-a~b-c']):
        path=tmp_path/f'{index}.body';path.write_bytes(gzip.compress(f'source_row_id,id,time\n{sid},{index},0.1\n'.encode()))
        previous,check=composer(path,source,writer,1,None,budget(tmp_path),encoded_gzip=True)
        assert previous is None and check['opaque_system_ids_unique']
    assert len(out.getvalue().splitlines())==2
    path=tmp_path/'duplicate';path.write_text('source_row_id,id,time\nrow-z.x_y,5,0.2\n')
    with pytest.raises(ValueError,match='duplicate opaque'):
        composer(path,source,writer,1,None,budget(tmp_path),encoded_gzip=False)
    composer.close()


def test_decoded_bomb_and_corrupt_gzip_stop_without_repair(tmp_path):
    source=dict(asset_id='fixture',expected_native_fields=['id','time']);path=tmp_path/'compressed'
    path.write_bytes(gzip.compress(b'source_row_id,id,time\nrow-a,1,0.1\n'))
    composer=NativeComposer(tmp_path,10)
    with pytest.raises(BudgetLimit,match='decoded'):composer(path,source,csv.writer(io.StringIO()),1,None,budget(tmp_path),encoded_gzip=True)
    composer.close()
    folder=tmp_path/'corrupt';folder.mkdir();composer=NativeComposer(folder,10000);path.write_bytes(b'not gzip')
    with pytest.raises(gzip.BadGzipFile):composer(path,source,csv.writer(io.StringIO()),1,None,budget(tmp_path),encoded_gzip=True)
    composer.close()


def test_compressed_fetch_retains_encoded_bytes_and_explicit_opt_in(tmp_path):
    body=gzip.compress(b'a,b\n1,2\n')
    class Response(io.BytesIO):
        status=200
        headers={'Content-Type':'text/csv','Content-Encoding':'gzip','Content-Length':str(len(body))}
        def geturl(self):return 'https://data.transportation.gov/resource/fixture.csv'
    class Opener:
        calls=0
        def open(self,request,**kwargs):
            self.calls+=1;assert request.get_header('Accept-encoding')=='gzip';return Response(body)
    opener=Opener();b=budget(tmp_path);path=tmp_path/'page.body'
    r=fetch('https://data.transportation.gov/resource/fixture.csv',path,b,tmp_path/'receipt.json',opener=opener,accept_gzip=True)
    assert opener.calls==1 and path.read_bytes()==body and b.download_bytes==len(body)
    assert r['encoded_wire_body_retained'] and gzip.decompress(path.read_bytes())==b'a,b\n1,2\n'


def test_remaining_caps_carry_parent_spend_without_reset(tmp_path,monkeypatch):
    import hashlib
    from scripts import recover_tgsim_paged_input as producer
    monkeypatch.setattr(producer,'ROOT',tmp_path)
    parent=tmp_path/'completion.json';parent.write_text(json.dumps(dict(status='failed_preserved_do_not_retry',rows_indexed=0,total_body_download_bytes_upper_bound=35296468,elapsed_seconds=383.804718084,http_requests=25,decoded_csv_bytes=122302692,two_task_body_upper_bound=222864504)))
    plan=dict(continuation_parent_completion=str(parent),continuation_parent_completion_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),prior_accounted_body_bound=222864504)
    caps=effective_caps(plan)
    assert caps['max_download_bytes_new_task']==501574444
    assert caps['maximum_http_requests']==119 and caps['max_wall_clock_seconds']==2316
    assert caps['max_decoded_csv_bytes']==1166187496
    plan['prior_accounted_body_bound']=0
    with pytest.raises(ValueError,match='bill not carried'):effective_caps(plan)


def test_read_timeout_is_explicit_and_cannot_exceed_hard_cap(tmp_path):
    class Response(io.BytesIO):
        status=200;headers={'Content-Type':'text/csv','Content-Length':'1'}
        def geturl(self):return 'https://data.transportation.gov/resource/fixture.csv'
    class Opener:
        seen=None
        def open(self,request,**kwargs):self.seen=kwargs['timeout'];return Response(b'x')
    b=budget(tmp_path);b.proposal['max_wall_clock_seconds']=200;opener=Opener()
    fetch('https://data.transportation.gov/resource/fixture.csv',tmp_path/'body',b,tmp_path/'receipt.json',opener=opener,read_timeout_seconds=120)
    assert opener.seen==120
    with pytest.raises(ValueError,match='bounded'):fetch('https://data.transportation.gov/resource/fixture.csv',tmp_path/'never',b,tmp_path/'never.json',opener=opener,read_timeout_seconds=121)
