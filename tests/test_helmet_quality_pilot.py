"""Four bounded producer cases, with no model/library load or network."""
from pathlib import Path
import json
import sys

import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_helmet_quality_pilot as q


def spec():
    return {'arm':'base','runtime':{},'generation':dict(q.GENERATION),'prompt':q.PROMPT,'resources':{},
            'case_order':['q%02d'%i for i in range(6)],'inputs':[{'case_id':'q%02d'%i,'crop_path':'/opaque/q%02d.png'%i,
            'crop_sha':'0'*64,'crop_rgb_sha':'0'*64,'width':32,'height':32} for i in range(6)],
            'source_commit':'0'*40,'source_sha':{},'caps':dict(q.CAPS)}


def test_native_rgb_crop_and_integer_gate(tmp_path):
    image=tmp_path/'source.png';Image.new('RGBA',(64,64),(10,20,30,0)).save(image)
    data,pixels,size=q.png_crop(image,[8.,9.,32.,33.])
    assert size==[32,33] and data.startswith(b'\x89PNG') and len(pixels)==64
    with pytest.raises(ValueError):q.png_crop(image,[8.2,9,32,33])


def test_label_free_worker_and_frozen_generation():
    v=spec();q.worker_schema(v)
    v['inputs'][0]['label']='HELMET'
    with pytest.raises(ValueError):q.worker_schema(v)
    v=spec();v['generation']['max_new_tokens']=33
    with pytest.raises(ValueError):q.worker_schema(v)


def test_failed_operation_is_counted_and_no_seventh_call(tmp_path):
    ledger=q.Ledger(tmp_path/'operations.jsonl','base')
    try:
        with pytest.raises(RuntimeError):
            with ledger.operation('generate','q00'):raise RuntimeError('synthetic')
        for i in range(5):
            with ledger.operation('generate','q%02d'%(i+1)):pass
        with pytest.raises(ValueError):
            with ledger.operation('generate','extra'):pass
    finally:ledger.close()
    rows=[json.loads(x) for x in (tmp_path/'operations.jsonl').read_text().splitlines()]
    assert len(rows)==12 and rows[0]['edge']=='begin' and rows[1]['status']=='FAIL'
    assert ledger.counts['generate']==6


def test_nonmodel_failed_child_retained_once(tmp_path):
    record=q.supervise([sys.executable,'-c','print("synthetic only");raise SystemExit(3)'],tmp_path/'child.log',10)
    assert record['returncode']==3 and not record['terminated_group']
    assert 'synthetic only' in (tmp_path/'child.log').read_text()
    assert record['finished']>=record['started']
