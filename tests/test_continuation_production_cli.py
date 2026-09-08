"""Actual production CLI refusals from an exact clean temporary code commit.

The repository is test-only; no production key/configuration or approval is installed.
Original scientific files and runs are never used as fixture outputs.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


@pytest.fixture
def committed_cli(tmp_path):
    repo=tmp_path/'implementation';repo.mkdir()
    files=[*sorted((ROOT/'scripts').glob('*continuation*.py')),ROOT/'src/runtime/fixed_commit_continuation.py']
    for source in files:
        dest=repo/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.PIPE).strip()
    git('init','-q');git('add','scripts','src')
    git('-c','user.name=Synthetic CLI fixture','-c','user.email=fixture@invalid',
        '-c','commit.gpgsign=false','commit','-qm','Test-only fixed source identity')
    identity={'version':'1.0.0','implementation_commit':git('rev-parse','HEAD'),
        'files':[{'path':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
                 for source in files]}
    assert git('status','--porcelain')==''
    run=tmp_path/'synthetic_run';run.mkdir()
    science=tmp_path/'synthetic_science';science.mkdir()
    spec=importlib.util.spec_from_file_location('external_security',ROOT/'scripts/continuation_executor_security.py')
    security=importlib.util.module_from_spec(spec);spec.loader.exec_module(security)
    contract={'version':'1.0.0','kind':'fixed_commit_continuation_execution_contract',
        'proposal_sha256':'a'*64,'proposal_file_sha256':'b'*64,
        'executor_identity_sha256':hashlib.sha256(canonical(identity)).hexdigest(),
        'run_id':run.name,'run_root':str(run),'scientific_worktree':str(science),
        'scientific_commit':'c'*40,'phases':list(security.PHASES),'write_scope':[str(run)],
        'ledger_anchors':[{},{}],'immutable_payload_inventory_sha256':'d'*64,
        'origin_evidence':{'launch_approval':{'status':'unavailable'},'release_attestation':{'status':'unavailable'}},
        'holdout_capability':False,'approval_domain':'production'}
    return repo,run,science,identity,contract,git


@pytest.mark.parametrize('case,reason',[
    ('missing_approval','independent continuation approval unavailable'),
    ('identity_hash','executor identity reference drift'),
    ('wrong_commit','executor commit drift'),
    ('dirty_source','executor tracked source dirty'),
    ('omitted_file','identity omits continuation'),
    ('file_hash','executor file drift'),
    ('synthetic_contract','synthetic approval cannot enter production CLI'),
    ('fake_approval',None),
    ('scope_escape','execution write scope'),
    ('run_mismatch','execution contract run identity'),
])
def test_actual_cli_fixed_identity_refusals_before_run_writes(tmp_path,committed_cli,case,reason,record_property):
    repo,run,science,identity,contract,git=committed_cli
    approval=None
    if case=='identity_hash':contract['executor_identity_sha256']='e'*64
    elif case=='wrong_commit':identity['implementation_commit']='e'*40
    elif case=='dirty_source':
        path=repo/'scripts/run_fixed_commit_continuation.py';path.write_text(path.read_text()+'\n# test-only dirty source\n')
    elif case=='omitted_file':identity['files']=[row for row in identity['files'] if row['path']!='scripts/continuation_qualification.py']
    elif case=='file_hash':identity['files'][0]['sha256']='e'*64
    elif case=='synthetic_contract':contract['approval_domain']='synthetic-only'
    elif case=='fake_approval':approval={'body':{'domain':'synthetic-only'},'key_id':'test-only','signature':'0'*64}
    elif case=='scope_escape':contract['write_scope']=[str(tmp_path)]
    elif case=='run_mismatch':contract['run_id']='different_run'
    if case!='identity_hash':contract['executor_identity_sha256']=hashlib.sha256(canonical(identity)).hexdigest()
    identity_path=tmp_path/'identity.json';identity_path.write_text(json.dumps(identity))
    contract_path=tmp_path/'contract.json';contract_path.write_text(json.dumps(contract))
    command=[sys.executable,'-I','-B',str(repo/'scripts/run_fixed_commit_continuation.py'),
        '--contract',str(contract_path),'--executor-identity',str(identity_path),'--phase','formal_controller']
    if approval:
        path=tmp_path/'fake_approval.json';path.write_text(json.dumps(approval))
        command.extend(['--approval',str(path),'--proposal',str(tmp_path/'missing_proposal.json')])
    result=subprocess.run(command,cwd=repo,text=True,capture_output=True)
    assert result.returncode==2,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report['status']=='denied' and report['execution_authorized'] is False
    assert report['lock_created'] is False and report['dispatch_started'] is False
    assert not list(run.iterdir()) and not list(science.iterdir())
    if reason:assert reason in report['reason'],report
    else:assert 'trust' in report['reason'] or 'signer' in report['reason'],report
    record_property('test_only_implementation_commit',identity['implementation_commit'])
    record_property('actual_cli_rejection',report['reason'])
