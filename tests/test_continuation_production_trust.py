"""Isolated tests invoke the production-shared Ed25519 core, without real trust."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.continuation_executor import PHASES
from scripts.continuation_executor.authorization import validate_contract, verify_approval
from scripts.continuation_executor.identity import ContinuationError, canonical, digest, file_hash, read_json
from scripts.continuation_executor.production_trust import TrustContext
from scripts.continuation_executor.test_trust_fixture import TestTrustFixture, signed, write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trust(tmp_path):
    run = tmp_path/'synthetic_trust'; run.mkdir()
    proposal = dict(run_id=run.name, run_root=str(run), ledgers=[])
    contract = dict(version='1.0.0', domain='synthetic', proposal_sha256=digest(proposal), proposal_file_sha256=digest(proposal),
        executor_identity_sha256=digest({}), run_id=run.name, run_root=str(run), phases=list(PHASES),
        holdout_capability=False, prefixes=[], immutable_files=[], fixture_root=str(tmp_path),
        expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), revocation_id='test-approval',
        recovery_owner_sha256=None, recovery_quiescence=None, coordination_root=str(tmp_path/'.continuation_locks'),
        command_plan_sha256=digest({}))
    t = TestTrustFixture(contract)
    validate_contract(contract, proposal, {})
    return t


def test_valid_shared_original_certificates(trust):
    result = trust.verify()
    assert result['approval_verified'] and result['real_execution_authorized'] is False
    assert len({r['message']['original']['sha256'] for r in trust.evidence.values()}) == 1
    assert result['revocation']['sequence'] == 0


@pytest.mark.parametrize('case', ['signature', 'record_signature', 'record_tamper', 'original_tamper',
    'author', 'run', 'commit', 'release', 'scope', 'coverage', 'verifier', 'time', 'missing_original', 'fake_state_hash'])
def test_certification_negative(trust, case):
    row = trust.evidence['release_attestation']
    if case == 'signature':
        trust.approval['signature'] = '00'*64
    elif case == 'original_tamper':
        trust.original.write_text('{}')
    elif case == 'missing_original':
        trust.original.unlink()
    elif case == 'fake_state_hash':
        trust.evidence['release_attestation'] = dict(state='independently_verified', reference_sha256='a'*64)
        trust.approve()
    else:
        if case == 'record_signature': row['signature'] = '00'*64
        elif case == 'record_tamper': row['message']['basis'] += 'altered'
        else:
            m = row['message']
            if case == 'author': m['original']['author'] = 'impostor'
            if case == 'run': m['scope']['run_id'] = 'synthetic_other'
            if case == 'commit': m['scope']['scientific_commit'] = 'b'*40
            if case == 'release': m['scope']['release_identity'] = 'other'
            if case == 'scope': m['scope']['phases'] += ['holdout']
            if case == 'coverage': m['coverage'] = {}
            if case == 'verifier': m['verifier_id'] = 'impostor'
            if case == 'time': m['verified_at'] = '1990-01-01T00:00:00+00:00'
            trust.evidence['release_attestation'] = signed(m, 'reviewer', trust.keys['reviewer'])
        trust.approve()
    with pytest.raises((ContinuationError, OSError)):
        trust.verify()


@pytest.mark.parametrize('case', ['missing', 'unreadable', 'signature', 'authority', 'source', 'scope',
    'expired', 'stale', 'future', 'negative_sequence', 'bool_sequence', 'revoked', 'signer', 'verifier'])
def test_revocation_fail_closed(trust, case):
    path = Path(trust.pin['revocation']['path'])
    if case == 'missing': path.unlink()
    elif case == 'unreadable': path.unlink(); path.mkdir()
    elif case == 'signature':
        row = read_json(path); row['signature'] = '00'*64; write(path, row)
    else:
        changes = {}
        if case == 'authority': changes['authority'] = 'other'
        if case == 'source': changes['source_id'] = 'other'
        if case == 'scope': changes['scope'] = {}
        if case == 'expired': changes['expires_at'] = trust.now.isoformat()
        if case == 'stale': changes['issued_at'] = (trust.now-timedelta(hours=3)).isoformat()
        if case == 'future': changes['issued_at'] = (trust.now+timedelta(hours=3)).isoformat()
        if case == 'negative_sequence': changes['sequence'] = -1
        if case == 'bool_sequence': changes['sequence'] = True
        if case == 'revoked': changes['revoked_ids'] = [trust.contract['revocation_id']]
        if case == 'signer': changes['untrusted_signers'] = ['approver']
        if case == 'verifier': changes['untrusted_signers'] = ['reviewer']
        trust.publish(**changes)
    with pytest.raises((ContinuationError, OSError)):
        trust.verify()


def test_revoked_state_cannot_roll_back_to_allowed(trust):
    trust.verify()
    trust.publish(sequence=2, revoked_ids=[trust.contract['revocation_id']])
    with pytest.raises(ContinuationError, match='revoked'): trust.verify()
    trust.restart()
    trust.publish(sequence=0)
    with pytest.raises(ContinuationError, match='rollback'): trust.verify()


def test_same_sequence_equivocation(trust):
    trust.verify()
    trust.publish(sequence=0, revoked_ids=['different'])
    with pytest.raises(ContinuationError, match='equivocation'): trust.verify()


def test_restart_requires_external_checkpoint(trust):
    initial_pin = deepcopy(trust.pin)
    trust.verify()
    trust.restart()
    assert trust.verify()['approval_verified']
    trust.context = TrustContext(initial_pin, test_only=True)
    with pytest.raises(ContinuationError, match='challenge mismatch'): trust.verify()


@pytest.mark.parametrize('case', ['missing', 'reset', 'clock_backward', 'clock_forward', 'lock_missing'])
def test_continuity_and_clock_fail_closed(trust, case):
    trust.verify()
    now = None
    if case == 'missing': trust.state.unlink()
    if case == 'reset': write(trust.state, trust.initial)
    if case == 'clock_backward': now = trust.now-timedelta(seconds=1)
    if case == 'clock_forward': now = trust.now+timedelta(hours=4)
    if case == 'lock_missing': Path(str(trust.state)+'.lock').unlink()
    with pytest.raises((ContinuationError, OSError)): trust.verify(now=now)


def test_scope_cannot_become_real_capability(trust):
    contract = deepcopy(trust.contract)
    contract.update(domain='production', fixture_root=None)
    with pytest.raises(ContinuationError, match='test approval'):
        verify_approval(contract, trust.approval, test_trust_context=trust.context)
    with pytest.raises(ContinuationError, match='production trust unavailable'):
        verify_approval(contract, trust.approval)
    trust.contract['run_root'] = '/Users/howen/Projects/PPO_MEC/artifacts/experiments/real'
    trust.approve()
    with pytest.raises(ContinuationError, match='scope'): trust.verify()


@pytest.mark.parametrize('when', ['before_lock', 'after_lock'])
def test_signed_revocation_at_lock_boundaries(trust, monkeypatch, when):
    import scripts.continuation_executor.locking as locking
    calls = []
    def authorize():
        calls.append(1)
        if when == 'before_lock' or len(calls) == 2:
            trust.publish(sequence=1, revoked_ids=[trust.contract['revocation_id']])
        return trust.verify()
    with pytest.raises(ContinuationError, match='revoked'):
        with locking.SingleWriter(trust.contract['run_root'], 'test', authorize):
            pytest.fail('revoked grant entered writer')
    lock = locking.writer_lock_path(trust.contract['run_root'])
    assert not lock.exists() if when == 'before_lock' else lock.read_bytes() == b''


@pytest.mark.parametrize('case', ['valid', 'owner_tamper', 'certificate_tamper', 'namespace_missing', 'false_quiescence'])
def test_quiescence_original_certification(trust, case):
    owner = dict(version='1.0.0', state='held', nonce='test', process=dict(host='test-host', pid=1, started='test'), executor_sha256='a'*64)
    trust.contract['recovery_owner_sha256'] = digest(owner)
    source = trust.root/'quiescence_test_only.json'
    body = dict(owner=owner, host='test-host', namespace='test namespace', observed_at=trust.now.isoformat(),
        no_live_descendants=True, basis='Synthetic observation only')
    if case == 'owner_tamper': body['owner']['nonce'] = 'tampered'
    if case == 'namespace_missing': body['namespace'] = ''
    if case == 'false_quiescence': body['no_live_descendants'] = False
    write(source, body)
    trust.contract['recovery_quiescence'] = trust.certify('recovery_quiescence', source)
    if case == 'certificate_tamper': trust.contract['recovery_quiescence']['message']['basis'] = 'tampered'
    trust.approve()
    if case == 'valid': assert trust.verify()['approval_verified']
    else:
        with pytest.raises(ContinuationError): trust.verify()


@pytest.mark.parametrize('domain', ['synthetic', 'production'])
@pytest.mark.parametrize('mode', ['qualification', 'execute'])
def test_public_cli_refuses_test_trust(trust, domain, mode):
    p = dict(run_id=trust.contract['run_id'], run_root=trust.contract['run_root'], ledgers=[])
    trust.contract.update(domain=domain, fixture_root=str(trust.root) if domain == 'synthetic' else None)
    write(trust.root/'proposal.json', p)
    trust.contract['proposal_file_sha256'] = file_hash(trust.root/'proposal.json')
    trust.approve()
    for name, value in [('contract', trust.contract), ('approval', trust.approval), ('identity', {})]: write(trust.root/(name+'.json'), value)
    before = {str(p): file_hash(p) for p in trust.root.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/execute_fixed_commit_continuation.py'),
        '--proposal', str(trust.root/'proposal.json'), '--contract', str(trust.root/'contract.json'),
        '--approval', str(trust.root/'approval.json'), '--executor-identity', str(trust.root/'identity.json'),
        '--phase', PHASES[0], '--check', mode, '--finalize-phase-only'], capture_output=True, text=True)
    assert result.returncode == 2 and json.loads(result.stdout)['execution_authorized'] is False
    assert before == {str(p): file_hash(p) for p in trust.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('case', ['exit75', 'terminal', 'missing', 'corrupt', 'descriptor', 'provenance',
    'publication_crash', 'candidate_crash', 'duplicate_committed', 'gate_missing', 'gate_false',
    'revoke_during_phase', 'revoke_retry', 'candidate_crash_revoked', 'expire_during_phase', 'utc_adjustment',
    'truncation', 'fork', 'out_of_order', 'cross_ledger', 'immutable_payload'])
def test_v2_native_transactions_and_restart(tmp_path, case):
    import cryptography
    dependencies = str(Path(cryptography.__file__).resolve().parent.parent)
    old = '/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847'
    result = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'tests/continuation_native_driver.py'),
        str(tmp_path), case, dependencies], cwd=old,
        env=dict(os.environ, PYTHONPATH=old, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'pass'
    assert report.get('prefix_unchanged', report.get('rejected_before_write'))


@pytest.mark.parametrize('case', ['installation_record', 'state_symlink', 'source_symlink', 'startup_missing', 'wrong_pin'])
def test_installation_boundary(trust, case):
    if case == 'installation_record': Path(trust.pin['installation_record']['path']).write_text('{}')
    if case == 'state_symlink':
        trust.state.unlink(); trust.state.symlink_to(trust.original)
    if case == 'source_symlink':
        p = Path(trust.pin['revocation']['path']); p.unlink(); p.symlink_to(trust.original)
    if case == 'startup_missing': trust.state.unlink()
    if case == 'wrong_pin': trust.context.pin['revocation']['public_key'] = '00'*32
    with pytest.raises((ContinuationError, OSError)): trust.verify()


@pytest.mark.parametrize('field,value', [('revoked_ids', 'test-approval'), ('untrusted_signers', 'approver'), ('untrusted_signers', 'revoker')])
def test_higher_sequence_cannot_erase_known_revocation(trust, field, value):
    trust.publish(sequence=1, **{field: [value]})
    with pytest.raises(ContinuationError): trust.verify()
    trust.restart()
    trust.publish(sequence=2)
    with pytest.raises(ContinuationError, match='set rollback'): trust.verify()


@pytest.mark.parametrize('case', ['rollback_to_bootstrap', 'old_receipt', 'bad_signature', 'stale_receipt'])
def test_startup_challenge_blocks_replayed_checkpoint(trust, case):
    old_receipt = read_json(trust.pin['startup_receipt_path'])
    trust.verify()
    trust.publish(sequence=2, revoked_ids=[trust.contract['revocation_id']])
    with pytest.raises(ContinuationError, match='revoked'): trust.verify()
    current_checkpoint = trust.context.expected
    trust.context = TrustContext(trust.pin, test_only=True)
    if case == 'rollback_to_bootstrap':
        write(trust.state, trust.initial)
        trust.publish(sequence=0)
        write(trust.pin['startup_receipt_path'], old_receipt)
    elif case == 'old_receipt': write(trust.pin['startup_receipt_path'], old_receipt)
    else:
        message = dict(trust.context.startup_request(), authority='revoker', checkpoint_sha256=current_checkpoint,
            issued_at=trust.now.isoformat(), expires_at=(trust.now+timedelta(hours=1)).isoformat())
        if case == 'stale_receipt': message['expires_at'] = trust.now.isoformat()
        receipt = signed(message, 'revoker', trust.keys['revoker'])
        if case == 'bad_signature': receipt['signature'] = '00'*64
        write(trust.pin['startup_receipt_path'], receipt)
    with pytest.raises(ContinuationError): trust.verify()


def test_fresh_custodian_receipt_detects_local_bootstrap_rollback(trust):
    trust.verify()
    trust.publish(sequence=2, revoked_ids=[trust.contract['revocation_id']])
    with pytest.raises(ContinuationError): trust.verify()
    trust.restart()  # Independent custodian retained the sequence-2 checkpoint.
    write(trust.state, trust.initial)
    trust.publish(sequence=0)
    with pytest.raises(ContinuationError, match='checkpoint mismatch'): trust.verify()


def test_missing_startup_receipt_never_bootstraps(trust):
    Path(trust.pin['startup_receipt_path']).unlink()
    with pytest.raises(OSError): trust.verify()
    assert read_json(trust.state) == trust.initial
