"""Synthetic trust and append tests; no real run or scientific execution."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from src.runtime import fixed_commit_continuation as c

ROOT = Path(__file__).resolve().parents[1]


def record(seq, previous, phase='checkpoint_freeze', kind='phase', status='completed'):
    cur, prev = ('current_record_hash', 'previous_record_hash') if kind == 'phase' else ('current_ledger_hash', 'previous_ledger_hash')
    r = dict(sequence_number=seq, phase=phase, status=status,
             run_identity_fingerprint='a'*64, failure_classification=None,
             cell_id='cell-' + str(seq), **{prev: previous})
    r[cur] = c.digest(r)
    return r


@pytest.fixture
def proposal(tmp_path):
    run = tmp_path / 'synthetic_continuation'
    run.mkdir()
    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(['git','init','-q',str(source)],check=True)
    (source/'source.py').write_text('# frozen\n')
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    evidence=[]
    for role in c.ROLES:
        f=run/(role+'.json');f.write_text('{}\n');evidence.append(c.identity(f,role))
    ledgers=[]
    for kind in ['phase','cell']:
        f=run/(kind+'_state.jsonl')
        f.write_text(json.dumps(record(1,None,kind=kind,status='completed' if kind=='phase' else 'committed'))+'\n')
        ledgers.append(c.ledger_anchor(f,kind))
    return dict(version=c.VERSION,state='proposal',execution_authorized=False,holdout_capability=False,
        run_id=run.name,run_root=str(run),worktree_root=str(source),execution_commit=c.git(source,'rev-parse','HEAD'),
        source_tree={'git_tree':c.git(source,'rev-parse','HEAD^{tree}'),'tracked_sources_sha256':'b'*64},
        evidence=evidence,ledgers=ledgers,allowed_phases=list(c.PHASES),
        write_scope={'root':str(run),'mode':'approved_append_only_successors','immutable_prefix':True,'currently_writable':False},
        origin_evidence={k:{'status':'unavailable','path':None,'sha256':None,'boundary':'not independently verified'} for k in ['launch_approval','release_attestation']},
        executor={'state':'pending','commit':None,'files':[]},
        approval={'state':'pending','source':None,'scope':'implementation_and_read_only_acceptance_only'},
        revocation_rules=['expired','revoked','identity_drift','failed_terminal','prefix_changed','unauthorized_successor'])


def signed(p,key=b'test-only-key'):
    a={'proposal_sha256':c.digest(p),'run_root':p['run_root'],'scope':list(c.PHASES),
       'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'revoked':False}
    a['signature']=hmac.new(key,c.canonical(a),hashlib.sha256).hexdigest()
    return a


def test_structure_and_main_progress_never_authorizes(proposal):
    assert c.validate_structure(proposal)['status']=='pass'
    for mode in ['structure','evidence','authorization']:
        assert c.preflight(proposal,mode)['execution_authorized'] is False
    # Main observation is intentionally not part of the immutable subject.
    source=Path(proposal['worktree_root'])
    (source/'source.py').write_text('# main advanced\n')
    result=c.preflight(proposal)
    assert not result['execution_authorized']
    assert any(r['status']=='fail' for r in result['checks'])


@pytest.mark.parametrize('field,value', [('execution_authorized',True),('holdout_capability',True),
 ('state','approved'),('execution_commit','HEAD'),('executor',{'state':'approved'}),
 ('approval',{'approved':True,'signer':'implementation agent'}),('run_id','second_run')])
def test_structure_escalation_rejected(proposal,field,value):
    proposal[field]=value
    with pytest.raises(ValueError):c.validate_structure(proposal)


@pytest.mark.parametrize('phase',['train','dev_select','checkpoint_freeze','holdout','second_run'])
def test_phase_scope_rejected(proposal,phase):
    proposal['allowed_phases'].append(phase)
    with pytest.raises(ValueError):c.validate_structure(proposal)


def test_fixture_approval_only(proposal,tmp_path):
    a=signed(proposal)
    assert c.check_fixture_approval(proposal,a,trusted_key=b'test-only-key',fixture_root=tmp_path)['fixture_approval_valid']
    proposal['run_id']=c.V16
    with pytest.raises(ValueError):c.check_fixture_approval(proposal,a,trusted_key=b'test-only-key',fixture_root=tmp_path)


@pytest.mark.parametrize('mutation',['forged','expired','revoked','scope','cross_run','proposal_hash'])
def test_invalid_approval(proposal,tmp_path,mutation):
    a=signed(proposal)
    if mutation=='forged':a['signature']=c.digest(a)
    else:
        if mutation=='expired':a['expires_at']='2000-01-01T00:00:00+00:00'
        if mutation=='revoked':a['revoked']=True
        if mutation=='scope':a['scope']=['holdout']
        if mutation=='cross_run':a['run_root']=str(tmp_path/'second_run')
        if mutation=='proposal_hash':a['proposal_sha256']='0'*64
        a['signature']=hmac.new(b'test-only-key',c.canonical({k:v for k,v in a.items() if k!='signature'}),hashlib.sha256).hexdigest()
    with pytest.raises(ValueError):c.check_fixture_approval(proposal,a,trusted_key=b'test-only-key',fixture_root=tmp_path)


@pytest.mark.parametrize('role',c.ROLES)
def test_evidence_drift(proposal,role):
    row=next(x for x in proposal['evidence'] if x['role']==role)
    Path(row['path']).write_text('{"changed":true}')
    r=c.preflight(proposal)
    assert any(x['name']=='file:'+role and x['status']=='fail' for x in r['checks'])
    assert not r['execution_authorized']


def append(anchor,phase='formal_cache_policy',kind='phase',**updates):
    path=Path(anchor['path']);rows=[json.loads(x) for x in path.read_text().splitlines()]
    cur='current_record_hash' if kind=='phase' else 'current_ledger_hash'
    r=record(len(rows)+1,rows[-1][cur],phase,kind,status='running')
    r.update(updates);r[cur]=c.digest({k:v for k,v in r.items() if k!=cur})
    with path.open('a') as f:f.write(json.dumps(r)+'\n')


def test_legal_tip_progress_and_no_approval_rejects(proposal):
    anchor=proposal['ledgers'][0]
    append(anchor)
    r=c.validate_ledger(anchor,c.PHASES)
    assert r['tip']!=anchor['terminal_hash'] and r['successor_count']==1
    with pytest.raises(ValueError):c.validate_ledger(anchor)


@pytest.mark.parametrize('attack',['prefix','truncate','unapproved','cross_run','fork','sequence','failed','skip','committed_replacement'])
def test_ledger_attacks(proposal,attack):
    a=proposal['ledgers'][0];path=Path(a['path'])
    if attack=='prefix':path.write_text(path.read_text().replace('checkpoint_freeze','train'))
    elif attack=='truncate':path.write_bytes(path.read_bytes()[:-2])
    elif attack=='unapproved':append(a,phase='holdout')
    elif attack=='cross_run':append(a,run_identity_fingerprint='b'*64)
    elif attack=='fork':append(a,previous_record_hash='b'*64)
    elif attack=='sequence':append(a,sequence_number=9)
    elif attack=='failed':append(a,status='failed')
    elif attack=='skip':append(a,phase='formal_support')
    elif attack=='committed_replacement':append(a,phase='checkpoint_freeze')
    with pytest.raises(ValueError):c.validate_ledger(a,c.PHASES)


def test_committed_cell_replacement(proposal):
    a=proposal['ledgers'][1]
    append(a,kind='cell',status='committed',cell_id='cell-1')
    with pytest.raises(ValueError):c.validate_ledger(a,c.PHASES)


@pytest.mark.parametrize('payload',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_strict_json(payload):
    with pytest.raises(ValueError):c.strict_json(payload)


def test_cli_rejects_before_any_child(proposal,tmp_path,monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('cli',ROOT/'scripts/preflight_fixed_commit_continuation.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    proposal['execution_authorized']=True
    path=tmp_path/'proposal.json';path.write_text(json.dumps(proposal))
    def forbidden(*a,**kw):raise AssertionError('child/dispatch attempted')
    monkeypatch.setattr(cli.subprocess,'run',forbidden)
    assert cli.main(['--proposal',str(path),'--check','authorization'])==2
    r=json.loads(capsys.readouterr().out)
    assert r['dispatch_count']==dict(train=0,dev=0,formal=0,holdout=0)


@pytest.mark.parametrize('shadow',['current_main','external'])
def test_cli_ignores_shadow_src(proposal,tmp_path,shadow):
    import os
    path=tmp_path/'proposal.json';path.write_text(json.dumps(proposal))
    shadow_root=ROOT if shadow=='current_main' else tmp_path/'shadow'
    if shadow=='external':
        (shadow_root/'src').mkdir(parents=True)
        (shadow_root/'src/__init__.py').write_text('raise RuntimeError("shadow import")')
    r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/preflight_fixed_commit_continuation.py'),
                      '--proposal',str(path),'--check','structure'],capture_output=True,text=True,
                      env=dict(os.environ,PYTHONPATH=str(shadow_root)))
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['execution_authorized'] is False


def test_public_gate_unchanged():
    runner=(ROOT/'scripts/run_typed_model_cache_formal_protocol.py').read_text()
    assert 'require_origin_main_match=False' not in runner
    import inspect
    from src.runtime.active_formal_bundle import validate_active_formal_bundle
    assert inspect.signature(validate_active_formal_bundle).parameters['require_origin_main_match'].default is True


def test_schema_required_fields_match_validator(proposal):
    schema=c.load(ROOT/'configs/experiment/fixed_commit_continuation_v1/continuation_schema.json')
    assert set(schema['required'])==set(proposal)
    for field in schema['required']:
        altered=deepcopy(proposal);altered.pop(field)
        with pytest.raises(ValueError):c.validate_structure(altered)


@pytest.mark.parametrize('outside',['current_main','external'])
def test_probe_rejects_preloaded_shadow(tmp_path,monkeypatch,outside):
    import types
    spec=importlib.util.spec_from_file_location('probe',ROOT/'scripts/probe_fixed_commit_continuation.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    fake=types.ModuleType('src.shadow')
    fake.__file__=str((ROOT if outside=='current_main' else tmp_path/'external')/'src/shadow.py')
    monkeypatch.setitem(sys.modules,'src.shadow',fake)
    with pytest.raises(ValueError):mod.validate_loaded_origins(tmp_path/'old_source')


def test_protected_inventory_and_committed_replacement(tmp_path):
    import gzip
    run=tmp_path/'synthetic_inventory';run.mkdir()
    dest=run/'cell';dest.mkdir();payload=dest/'result.json';payload.write_text('{}')
    rows=[{'path':'result.json','sha256':c.file_hash(payload),'size_bytes':2}]
    r={'status':'committed','committed_path':str(dest),'artifact_inventory':rows,
       'artifact_inventory_sha256':c.digest(rows),'cell_id':'c','run_identity_fingerprint':'a'*64,
       'command_hash':'b'*64,'input_hash':'c'*64,'cell_artifact_publication_contract_version':'1.0.0'}
    (dest/'committed_marker.json').write_text(json.dumps({**r,'committed_destination':str(dest)}))
    (run/'cell_state.jsonl').write_text(json.dumps(r)+'\n')
    (run/'phase_state.jsonl').write_text('')
    inv=tmp_path/'inventory.json.gz'
    def snapshot():
        files=[{k:v for k,v in c.identity(p,'resource').items() if k!='role'} for p in run.rglob('*') if p.is_file()]
        with gzip.open(inv,'wt') as f:json.dump({'files':files},f)
    snapshot()
    assert c.validate_protected_inventory(inv,immutable_roots=[str(run)])['status']=='pass'
    assert c.validate_committed_payloads(run,inv)['committed_cells']==1
    payload.write_text('{"replacement":true}')
    with pytest.raises(ValueError):c.validate_protected_inventory(inv)
    snapshot()  # Even a freshly self-hashed replacement must disagree with the committed ledger.
    with pytest.raises(ValueError):c.validate_committed_payloads(run,inv)


def test_cli_missing_protected_inventory_cannot_skip_check(proposal,tmp_path,monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('cli_missing',ROOT/'scripts/preflight_fixed_commit_continuation.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    path=tmp_path/'proposal.json';path.write_text(json.dumps(proposal))
    original=cli.subprocess.run
    def restricted(*args,**kwargs):
        if args and args[0][0]=='git':return original(*args,**kwargs)
        raise AssertionError('scientific child must not be called')
    monkeypatch.setattr(cli.subprocess,'run',restricted)
    assert cli.main(['--proposal',str(path)])==2
    assert 'complete protected JSON gzip inventory' in json.loads(capsys.readouterr().out)['error']
