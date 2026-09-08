"""Explicit test-only dispatch authority. Never read by production admission."""
from __future__ import annotations
import hashlib
import hmac
import importlib.util
import json
import secrets
import time
from pathlib import Path


def security():
    spec=importlib.util.spec_from_file_location('fixture_authorization_security',
        Path(__file__).with_name('continuation_executor_security.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def issue(fixture,run,implementation,*,parent_approval_sha256=None,expires_at=None):
    sec=security();fixture=sec.validate_fixture_root(fixture)
    run=sec.contained(run,fixture,must_exist=True);implementation=Path(implementation)
    if not run.name.startswith('synthetic_'):raise ValueError('test-only synthetic run required')
    body={'version':'1.0.0','domain':'synthetic-only','kind':'fixture_dispatch_scope',
        'fixture_root':str(fixture),'run_root':str(run),'run_id':run.name,
        'phases':list(sec.PHASES),'issued_at':time.time()-1,
        'expires_at':time.time()+3600 if expires_at is None else expires_at,
        'parent_approval_sha256':parent_approval_sha256,'revoked':False,
        'executor_files':[{'path':str(path),'sha256':sec.file_hash(path)}
            for path in sorted((implementation/'scripts').glob('*continuation*.py'))]}
    key=secrets.token_bytes(32)
    payload={'body':body,'test_only_key_hex':key.hex(),
        'signature':hmac.new(key,sec.canonical(body),hashlib.sha256).hexdigest()}
    path=fixture/'fixture_dispatch_scope.json'
    with path.open('x') as stream:json.dump(payload,stream,allow_nan=False)
    return {'path':str(path),'sha256':sec.file_hash(path),'domain':'synthetic-only'}


def verify(fixture,implementation,*,phase,output_paths,cell_output_root=None,now=None):
    sec=security();fixture=sec.validate_fixture_root(fixture)
    path=sec.contained(fixture/'fixture_dispatch_scope.json',fixture,must_exist=True)
    payload=sec.strict_json(path)
    if set(payload)!={'body','test_only_key_hex','signature'}:raise ValueError('fixture authority schema')
    body=payload['body'];key=bytes.fromhex(payload['test_only_key_hex'])
    if len(key)!=32 or not hmac.compare_digest(hmac.new(key,sec.canonical(body),hashlib.sha256).hexdigest(),payload['signature']):
        raise ValueError('fixture authority signature')
    if body.get('domain')!='synthetic-only' or body.get('kind')!='fixture_dispatch_scope' or body.get('version')!='1.0.0':
        raise ValueError('fixture authority domain')
    if body.get('fixture_root')!=str(fixture) or body.get('revoked') is not False:
        raise ValueError('fixture authority root/revocation')
    now=time.time() if now is None else now
    if not body['issued_at']<=now<body['expires_at']:raise ValueError('fixture authority expired/not yet valid')
    if body['phases']!=list(sec.PHASES) or phase not in body['phases']:raise ValueError('fixture authority phase')
    run=sec.contained(body['run_root'],fixture,must_exist=True)
    if run.name!=body['run_id'] or not run.name.startswith('synthetic_'):raise ValueError('fixture authority cross-run')
    expected={str(path) for path in (Path(implementation)/'scripts').glob('*continuation*.py')}
    rows=body['executor_files']
    if len(rows)!=len(expected) or {row['path'] for row in rows}!=expected:raise ValueError('fixture executor inventory drift')
    for row in rows:
        source=sec.contained(row['path'],implementation,must_exist=True)
        if sec.file_hash(source)!=row['sha256']:raise ValueError('fixture executor source drift')
    for output in output_paths:
        if output:sec.contained(output,run)
    cell_attempt=None
    if cell_output_root is not None:
        output=Path(cell_output_root)
        rows=[json.loads(line) for line in (run/'cell_state.jsonl').read_text().splitlines()]
        last={row['cell_id']:row for row in rows}
        candidates=[]
        for row in last.values():
            stage=Path(row['staging_path'])
            expected_output=stage/'artifact/benchmark' if phase=='formal_cache_policy' else stage/'child_output'
            if row['phase']==phase and row['status']=='running' and output==expected_output:
                candidates.append(row)
        if len(candidates)!=1:raise ValueError('fixture output not bound to an active phase/cell staging')
        cell_attempt=candidates[0]['attempt']
    return {'domain':'synthetic-only','phase':phase,'run_id':run.name,
            'authority_sha256':sec.file_hash(path),'expires_at':body['expires_at'],'cell_attempt':cell_attempt}
