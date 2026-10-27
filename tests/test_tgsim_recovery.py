"""Native opaque IDs, encoded-wire accounting and bounded decompression."""
import csv
import gzip
import io
import json

import pytest

from scripts.prepare_tgsim_input import Budget, BudgetLimit, fetch
from scripts.prepare_tgsim_paged_input import system_id_key
from scripts.recover_tgsim_paged_input import NativeComposer


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
