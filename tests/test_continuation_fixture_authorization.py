from __future__ import annotations
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCIENCE=Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')


@pytest.mark.parametrize('case,reason',[
    ('missing','missing path'),('expired','expired'),('revoked','root/revocation'),
    ('cross_run','outside approved root'),('executor_drift','source drift'),
    ('tamper','signature'),('phase','phase denied'),('output_scope','outside approved root'),
])
def test_actual_child_refuses_invalid_authority_before_any_write(tmp_path,case,reason):
    fixture=tmp_path/'synthetic_fixture';run=fixture/'synthetic_run';run.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('fixture_auth',ROOT/'scripts/continuation_fixture_authorization.py')
    authority=importlib.util.module_from_spec(spec);spec.loader.exec_module(authority)
    token=fixture/'fixture_dispatch_scope.json'
    if case!='missing':
        authority.issue(fixture,run,ROOT)
        payload=json.loads(token.read_text());body=payload['body']
        if case=='expired':body['expires_at']=time.time()-10
        elif case=='revoked':body['revoked']=True
        elif case=='cross_run':body['run_root']=str(tmp_path/'another_run')
        elif case=='executor_drift':body['executor_files'][0]['sha256']='0'*64
        elif case=='tamper':body['run_id']='forged'
        if case!='tamper':
            payload['signature']=hmac.new(bytes.fromhex(payload['test_only_key_hex']),
                authority.security().canonical(body),hashlib.sha256).hexdigest()
        token.write_text(json.dumps(payload))
    output=fixture/'outside_run' if case=='output_scope' else run/'staging/child_output'
    command=[sys.executable,'-I','-B',str(ROOT/'scripts/continuation_fixture_child.py'),
        '--fixture-root',str(fixture),'--scientific-root',str(SCIENCE),'--output-root',str(output),
        '--cell-phase','train' if case=='phase' else 'formal_ablation','--setting','synthetic_setting']
    before={str(path):path.read_bytes() for path in fixture.rglob('*') if path.is_file()}
    result=subprocess.run(command,cwd=SCIENCE,env=dict(os.environ,PYTHONPATH=str(SCIENCE)),text=True,capture_output=True)
    assert result.returncode!=0
    assert reason in result.stderr,result.stderr
    assert {str(path):path.read_bytes() for path in fixture.rglob('*') if path.is_file()}==before
    assert not output.exists()
