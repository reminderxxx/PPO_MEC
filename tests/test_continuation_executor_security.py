from __future__ import annotations

import hashlib
import hmac
import importlib.util
import multiprocessing
from pathlib import Path
import time

import pytest

_PATH = Path(__file__).resolve().parents[1] / 'scripts/continuation_executor_security.py'
_SPEC = importlib.util.spec_from_file_location('external_security', _PATH)
security = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(security)


def fixture(tmp_path):
    root = tmp_path / 'synthetic_fixture'
    run = root / 'synthetic_run'
    run.mkdir(parents=True)
    contract = {'executor_identity_sha256': 'a' * 64, 'proposal_sha256': 'b' * 64,
                'run_id': run.name, 'run_root': str(run),
                'phases': list(security.PHASES), 'write_scope': [str(run)]}
    trust = security.FixtureTrust(root, b'k' * 32, 'a' * 64)
    body = security.approval_body(contract, approval_id='fixture-approval',
                                  issued_at=10, expires_at=20, domain='synthetic-only')
    approval = {'body': body, 'key_id': 'test-only',
                'signature': hmac.new(trust.key, security.canonical(body), hashlib.sha256).hexdigest()}
    return run, contract, trust, approval


def check(contract, trust, approval, **kwargs):
    return security.verify_approval(contract, approval, phase=security.PHASES[0],
                                    domain='synthetic-only', verify_signature=trust.verify,
                                    revoked_ids=[], now=15, **kwargs)


def test_fixture_signature_cannot_be_production_trust(tmp_path):
    _, contract, trust, approval = fixture(tmp_path)
    assert check(contract, trust, approval)['domain'] == 'synthetic-only'
    with pytest.raises(security.AdmissionError, match='domain'):
        security.verify_approval(contract, approval, phase=security.PHASES[0],
                                 domain='production', verify_signature=trust.verify,
                                 revoked_ids=[], now=15)
    assert security.ProductionTrust({}, []).verify('test-only', b'body', 'signature') is False


@pytest.mark.parametrize('phase', ['train', 'dev_select', 'checkpoint_freeze', 'holdout', 'unknown'])
def test_forbidden_phase_even_with_signature(tmp_path, phase):
    _, contract, trust, approval = fixture(tmp_path)
    with pytest.raises(security.AdmissionError, match='phase'):
        security.verify_approval(contract, approval, phase=phase, domain='synthetic-only',
                                 verify_signature=trust.verify, revoked_ids=[], now=15)


@pytest.mark.parametrize('now,revoked', [(9, []), (20, []), (15, ['fixture-approval'])])
def test_expiry_and_revocation(tmp_path, now, revoked):
    _, contract, trust, approval = fixture(tmp_path)
    with pytest.raises(security.AdmissionError):
        security.verify_approval(contract, approval, phase=security.PHASES[0],
                                 domain='synthetic-only', verify_signature=trust.verify,
                                 revoked_ids=revoked, now=now)


@pytest.mark.parametrize('key,value', [('run_id', 'synthetic_other'), ('proposal_sha256', 'c'*64),
                                      ('write_scope', ['/real/run']), ('executor_identity_sha256', 'd'*64)])
def test_identity_scope_mutation(tmp_path, key, value):
    _, contract, trust, approval = fixture(tmp_path)
    contract[key] = value
    with pytest.raises(security.AdmissionError):
        check(contract, trust, approval)


def test_missing_approval_rejects_before_lock_creation(tmp_path):
    run, contract, trust, _ = fixture(tmp_path)
    with pytest.raises(security.AdmissionError, match='unavailable'):
        with security.single_writer(run, admission=lambda: check(contract, trust, None), owner_identity={}):
            pytest.fail('entered writer')
    assert list(run.iterdir()) == []


def test_recursive_reference_and_symlink_rejections(tmp_path):
    run, contract, trust, _ = fixture(tmp_path)
    trust.validate_contract(contract)
    trust.validate_references({'nested': ['--resource=' + str(run / 'resource.json')]})
    (run / 'escape').symlink_to(tmp_path)
    for value in [str(run / 'escape' / 'weight.pt'), '/real/v16/weight.pt',
                  '--resource=/real/data.csv', '../outside', 'file:///real/checkpoint',
                  {'nested': {'argv': ['/real/results.json']}}]:
        with pytest.raises(security.AdmissionError):
            trust.validate_references(value)
    contract['run_root'] = '/real/v16'
    with pytest.raises(security.AdmissionError):
        trust.validate_contract(contract)


def _hold_lock(root, ready):
    with security.single_writer(root, admission=lambda: {'valid': True}, owner_identity={'id': 'child'}):
        ready.send(True)
        time.sleep(30)


def test_concurrent_writer_and_kernel_crash_recovery(tmp_path):
    run, _, _, _ = fixture(tmp_path)
    context = multiprocessing.get_context('fork')
    parent, child = context.Pipe()
    process = context.Process(target=_hold_lock, args=(run, child))
    process.start()
    try:
        assert parent.poll(5) and parent.recv() is True
        inode = (run / '.continuation.lock').stat().st_ino
        with pytest.raises(security.AdmissionError, match='already active'):
            with security.single_writer(run, admission=lambda: {}, owner_identity={}):
                pytest.fail('second writer entered')
        process.terminate()
        process.join(5)
        assert not process.is_alive()
        with security.single_writer(run, admission=lambda: {}, owner_identity={'id': 'child'}):
            assert (run / '.continuation.lock').stat().st_ino == inode
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        child.close()


def test_revalidate_after_lock_before_owner_write(tmp_path):
    run, _, _, _ = fixture(tmp_path)
    calls = []
    def admission():
        calls.append(1)
        if len(calls) == 2:
            raise security.AdmissionError('revoked during acquisition')
        return {}
    with pytest.raises(security.AdmissionError, match='revoked'):
        with security.single_writer(run, admission=admission, owner_identity={}):
            pytest.fail('revoked writer entered')
    assert (run / '.continuation.lock').read_bytes() == b''
    with security.single_writer(run, admission=lambda: {}, owner_identity={}):
        pass


def test_strict_json(tmp_path):
    path = tmp_path / 'value.json'
    for text in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}']:
        path.write_text(text)
        with pytest.raises(security.AdmissionError):
            security.strict_json(path)


def test_production_trust_is_not_a_caller_owned_fixture(tmp_path, monkeypatch):
    path = tmp_path / 'self_approved_trust.json'
    path.write_text('{}')
    monkeypatch.setattr(security, 'PRODUCTION_TRUST_PATH', path)
    with pytest.raises(security.AdmissionError, match='root-owned'):
        security.load_production_trust()


def test_origin_status_cannot_be_replaced_by_signature_test(tmp_path):
    contract = {'origin_evidence': {'launch_approval': {'status': 'unavailable'},
                                   'release_attestation': {'status': 'unavailable'}}}
    with pytest.raises(security.AdmissionError, match='launch_approval unavailable'):
        security.verify_origin_attestations(contract, security.ProductionTrust({}, []))


@pytest.mark.parametrize('phase', ['train', 'dev_select', 'checkpoint_freeze', 'holdout', 'unknown'])
def test_actual_production_cli_phase_refusal_before_writes(tmp_path, phase):
    import subprocess
    import sys
    root = _PATH.parents[1]
    before = list(tmp_path.iterdir())
    result = subprocess.run([sys.executable, '-I', '-B', str(root / 'scripts/run_fixed_commit_continuation.py'),
                             '--contract', str(tmp_path / 'missing_contract.json'),
                             '--executor-identity', str(tmp_path / 'missing_identity.json'), '--phase', phase],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert 'unauthorized phase' in result.stdout
    assert list(tmp_path.iterdir()) == before


def test_fixture_rejects_source_and_real_locations_before_write(tmp_path):
    fake_source = tmp_path / 'old_source'
    fake_source.mkdir()
    (fake_source / '.git').write_text('gitdir: fixture-only')
    for path in (fake_source / 'synthetic_escape',
                 Path('/Users/howen/Projects/PPO_MEC/artifacts/experiments/synthetic_escape'),
                 Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847/synthetic_escape')):
        with pytest.raises(security.AdmissionError):
            security.validate_fixture_root(path)
        assert not path.exists()


def test_repeated_lock_keeps_integrity_bytes_and_rejects_identity_drift(tmp_path):
    run, _, _, _ = fixture(tmp_path)
    with security.single_writer(run, admission=lambda: {}, owner_identity={'executor': 'fixed'}):
        before = (run / '.continuation.lock').read_bytes()
    with security.single_writer(run, admission=lambda: {}, owner_identity={'executor': 'fixed'}):
        assert (run / '.continuation.lock').read_bytes() == before
    with pytest.raises(security.AdmissionError, match='identity drift'):
        with security.single_writer(run, admission=lambda: {}, owner_identity={'executor': 'other'}):
            pytest.fail('drift accepted')
    assert (run / '.continuation.lock').read_bytes() == before


def test_ed25519_signature_primitive_with_nonapproval_test_message(tmp_path,monkeypatch):
    import base64,json,subprocess
    node=Path('/Users/howen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    assert node.is_file(), 'acceptance requires the pinned external signature backend'
    program="""const c=require('node:crypto');const k=c.generateKeyPairSync('ed25519');
const m=Buffer.from('synthetic test message: not an approval or attestation');
process.stdout.write(JSON.stringify({public_key:k.publicKey.export({format:'der',type:'spki'}).subarray(-32).toString('base64'),
signature:c.sign(null,m,k.privateKey).toString('base64')}));"""
    pair=json.loads(subprocess.check_output([str(node),'-e',program],text=True,env={'LANG':'C'}))
    message=b'synthetic test message: not an approval or attestation'
    backend={'kind':'node_ed25519_v1','executable':str(node),'sha256':hashlib.sha256(node.read_bytes()).hexdigest()}
    trust=security.ProductionTrust({'test-key-in-memory-only':base64.b64decode(pair['public_key'])},[],backend)
    flag=tmp_path/'unwanted_preload';preload=tmp_path/'preload.cjs'
    preload.write_text('require("node:fs").writeFileSync('+json.dumps(str(flag))+',"executed")')
    monkeypatch.setenv('NODE_OPTIONS','--require '+str(preload))
    assert trust.verify('test-key-in-memory-only',message,pair['signature']) is True
    assert trust.verify('unknown-key',message,pair['signature']) is False
    assert trust.verify('test-key-in-memory-only',message+b'changed',pair['signature']) is False
    assert not flag.exists()
    trust.signature_backend={**backend,'sha256':'0'*64}
    assert trust.verify('test-key-in-memory-only',message,pair['signature']) is False
