from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('payload',[None,{}, {'version':'wrong'}, {'version':'1.0.0','approval_domain':'synthetic-only'}])
def test_actual_readonly_cli_rejects_missing_or_malformed_contract_without_writes(tmp_path,payload):
    contract=tmp_path/'invalid_contract.json'
    if payload is not None:contract.write_text(json.dumps(payload))
    before={str(path):path.read_bytes() for path in tmp_path.iterdir()}
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/inspect_fixed_commit_continuation.py'),
        '--proposal',str(tmp_path/'missing_proposal.json'),'--contract',str(contract)],
        cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==2,result.stderr
    report=json.loads(result.stdout)
    assert report['status']=='rejected' and report['execution_authorized'] is False
    assert {str(path):path.read_bytes() for path in tmp_path.iterdir()}==before


def test_protected_inventory_only_allows_anchored_suffix(tmp_path):
    import gzip,hashlib,importlib.util
    spec=importlib.util.spec_from_file_location('external_ledger_validation',ROOT/'scripts/continuation_ledger_validation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    science=tmp_path/'science';science.mkdir()
    source=science/'source.py';source.write_bytes(b'original source')
    ledger=tmp_path/'phase_state.jsonl';ledger.write_bytes(b'original-prefix\n')
    rows=[{'path':str(path),'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in [source,ledger]]
    anchor={'path':str(ledger),'prefix_sha256':rows[1]['sha256'],'byte_count':rows[1]['size_bytes']}
    inventory=tmp_path/'protected.json.gz'
    with gzip.open(inventory,'wt') as stream:json.dump({'files':rows},stream)
    ledger.write_bytes(ledger.read_bytes()+b'legal suffix checked by native ledger validator\n')
    result=module.validate_preserved_inventory(inventory,anchors=[anchor],scientific_root=science)
    assert result['verified_entries']==2
    source.write_bytes(b'changed source!')
    with pytest.raises(ValueError,match='drift'):
        module.validate_preserved_inventory(inventory,anchors=[anchor],scientific_root=science)
    source.write_bytes(b'original source')
    (science/'shadow.py').write_text('new')
    with pytest.raises(ValueError,match='new scientific'):
        module.validate_preserved_inventory(inventory,anchors=[anchor],scientific_root=science)


@pytest.mark.parametrize('case,reason',[
    ('wrong_interpreter','interpreter differs'),('wrong_cwd','cwd differs'),
    ('polluted_pythonpath','PYTHONPATH differs'),('user_site_enabled','user-site isolation'),
])
def test_actual_readonly_cli_rejects_environment_before_scientific_loading(tmp_path,case,reason):
    import hashlib,os
    proposal_path=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json')
    proposal=json.loads(proposal_path.read_text())
    canonical=lambda value:json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    contract={'version':'1.0.0','kind':'fixed_commit_continuation_execution_contract',
        'proposal_sha256':hashlib.sha256(canonical(proposal)).hexdigest(),
        'proposal_file_sha256':hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        'executor_identity_sha256':'0'*64,'run_id':proposal['run_id'],'run_root':proposal['run_root'],
        'scientific_worktree':proposal['worktree_root'],'scientific_commit':proposal['execution_commit'],
        'phases':proposal['allowed_phases'],'write_scope':[proposal['run_root']],
        'ledger_anchors':proposal['ledgers'],'immutable_payload_inventory_sha256':next(row['sha256'] for row in proposal['evidence'] if row['role']=='committed_output'),
        'origin_evidence':proposal['origin_evidence'],'holdout_capability':False,'approval_domain':'production'}
    path=tmp_path/'readonly_candidate.json';path.write_text(json.dumps(contract))
    env=dict(os.environ,PYTHONPATH=proposal['worktree_root'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0')
    if case=='polluted_pythonpath':env['PYTHONPATH']=str(ROOT)
    if case=='user_site_enabled':env.pop('PYTHONNOUSERSITE',None)
    interpreter='/usr/bin/python3' if case=='wrong_interpreter' else sys.executable
    result=subprocess.run([interpreter,'-I','-B',str(ROOT/'scripts/inspect_fixed_commit_continuation.py'),
        '--proposal',str(proposal_path),'--contract',str(path)],
        cwd=tmp_path if case=='wrong_cwd' else proposal['worktree_root'],env=env,text=True,capture_output=True)
    assert result.returncode==2,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert reason in report['error']
    assert report['execution_authorized'] is False
    assert report['counters']['scientific_calls']==0
    assert report['counters']['readonly_subprocesses']==[]
