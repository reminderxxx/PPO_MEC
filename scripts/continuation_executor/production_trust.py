"""Pinned Ed25519 certification and revocation core; no trust discovery.

The installation and restart checkpoint come from the trust owner, never the
approval. Test installations are confined to their synthetic root. This module
cannot establish the real-world truth of a signed certification or wall clock.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import re
import threading
import uuid

from . import PHASES
from .identity import (ContinuationError, absolute_path, canonical, digest,
                       read_json, strict_json, verify_file, within)

ROLES = ('launch_approval', 'release_attestation', 'continuation_approval')
# A future installation is a reviewed source change with a new executor identity.
PRODUCTION_INSTALLATION = None
_PRODUCTION_CONTEXT = None


def require(ok, reason):
    if not ok:
        raise ContinuationError(reason)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def utc(value):
    try:
        result = datetime.fromisoformat(value)
        require(result.tzinfo is not None, 'trusted time requires timezone')
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise ContinuationError('invalid trusted time') from exc


def verify_signature(envelope, signers):
    require(isinstance(envelope, dict) and set(envelope) == {'message', 'signer_id', 'signature'},
            'signed envelope schema')
    key = signers.get(envelope['signer_id'])
    require(key is not None, 'untrusted signer')
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(
            bytes.fromhex(envelope['signature']), canonical(envelope['message']))
    except Exception as exc:
        raise ContinuationError('invalid Ed25519 signature') from exc
    return envelope['message']


def scope_for(contract):
    return {k: contract[k] for k in ('run_id', 'run_root', 'scientific_commit',
            'release_identity', 'executor_identity_sha256', 'phases')}


class TrustContext:
    """An installed context plus in-process continuity. No auto bootstrap.

    Reopening requires a separately held exact checkpoint. A changed local state
    cannot silently become a new root of trust, even if it has a higher sequence.
    """
    def __init__(self, installation, *, test_only=False):
        self.pin = deepcopy(installation)
        p = self.pin
        require(set(p) == {'version', 'domain', 'fixture_root', 'installation_id',
            'trust_owner', 'installation_record', 'scope', 'approval_signers',
            'verifiers', 'revocation', 'continuity_path', 'startup_checkpoint_sha256'},
            'trust installation schema')
        require(p['version'] == '2.0.0' and text(p['installation_id']) and text(p['trust_owner']),
                'trust installation identity')
        require(p['domain'] == ('synthetic' if test_only else 'production'), 'test trust domain')
        require(test_only or p['fixture_root'] is None, 'production fixture forbidden')
        require(sha(p['startup_checkpoint_sha256']), 'external startup checkpoint required')
        verify_file(p['installation_record'])
        require(isinstance(p['approval_signers'], dict) and p['approval_signers'], 'approval pins missing')
        require(isinstance(p['verifiers'], dict) and p['verifiers'], 'verifier pins missing')
        for key in p['approval_signers'].values():
            require(sha(key), 'invalid public key pin')
        for name, v in p['verifiers'].items():
            require(text(name) and set(v) == {'public_key', 'author', 'roles'} and sha(v['public_key'])
                    and text(v['author']) and isinstance(v['roles'], list) and v['roles'], 'verifier pin schema')
        r = p['revocation']
        require(set(r) == {'authority', 'public_key', 'source_id', 'path', 'max_age_seconds'}
                and all(text(r[k]) for k in ('authority', 'source_id')) and sha(r['public_key'])
                and type(r['max_age_seconds']) is int and r['max_age_seconds'] > 0, 'revocation pin schema')
        require(isinstance(p['scope'], dict) and set(p['scope']) == {'run_id', 'run_root', 'scientific_commit',
                'release_identity', 'executor_identity_sha256', 'phases'}
                and p['scope']['phases'] == list(PHASES)
                and p['scope']['scientific_commit'] == 'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d'
                and sha(p['scope']['executor_identity_sha256']) and text(p['scope']['release_identity']),
                'installed continuation scope schema')
        root = absolute_path(p['scope']['run_root'])
        require(root.name == p['scope']['run_id'], 'installed run mismatch')
        self.state_path = within(p['continuity_path'], root.parent / '.continuation_locks')
        require(self.state_path.parent == root.parent / '.continuation_locks', 'continuity location')
        absolute_path(r['path'])
        if test_only:
            fixture = absolute_path(p['fixture_root'])
            require(root != fixture and root.name.startswith('synthetic_'), 'test run required')
            for path in (str(root), r['path'], str(self.state_path), p['installation_record']['path']):
                within(path, fixture)
        self.expected = p['startup_checkpoint_sha256']
        self.mutex = threading.Lock()

    def _scope(self, contract):
        require(contract['holdout_capability'] is False and contract['phases'] == list(PHASES), 'unauthorized phase/holdout')
        require(contract['version'] == '2.0.0', 'production trust requires contract 2.0.0')
        require(contract['domain'] == self.pin['domain'] and
                contract['fixture_root'] == self.pin['fixture_root'], 'installed domain/root mismatch')
        require(contract['trust_installation_id'] == self.pin['installation_id'] and
                scope_for(contract) == self.pin['scope'], 'installed scope mismatch')
        require(contract['coordination_root'] == str(self.state_path.parent), 'coordination mismatch')

    def _evidence(self, envelope, role, contract, now):
        m = verify_signature(envelope, {k: v['public_key'] for k, v in self.pin['verifiers'].items()})
        require(isinstance(m, dict) and set(m) == {'version', 'record_id', 'verifier_id', 'verified_at',
            'basis', 'original', 'scope', 'roles', 'coverage'}, 'certification schema')
        v = self.pin['verifiers'][envelope['signer_id']]
        require(m['version'] == '2.0.0' and m['verifier_id'] == envelope['signer_id']
                and text(m['record_id']) and text(m['basis']), 'certification identity/basis')
        require(isinstance(m['roles'], list) and role in m['roles'] and
                len(set(m['roles'])) == len(m['roles']) and set(m['roles']) <= set(v['roles']), 'certification role')
        require(isinstance(m['coverage'], dict) and set(m['coverage']) == set(m['roles'])
                and all(text(x) for x in m['coverage'].values()), 'certification coverage')
        expected = scope_for(contract)
        if role == 'launch_approval':
            expected['phases'] = ['preflight', 'tests', 'train', 'dev_select', 'checkpoint_freeze']
        if role == 'recovery_quiescence':
            expected['recovery_owner_sha256'] = contract['recovery_owner_sha256']
        require(m['scope'] == expected, 'certification scope mismatch')
        original = m['original']
        require(isinstance(original, dict) and set(original) == {'path', 'sha256', 'size_bytes',
            'author', 'recorded_at', 'source'}, 'original reference schema')
        require(sha(original['sha256']) and type(original['size_bytes']) is int and original['size_bytes'] >= 0, 'original byte identity')
        require(original['author'] == v['author'] and text(original['source']), 'original subject/source mismatch')
        require(utc(original['recorded_at']) <= utc(m['verified_at']) <= now, 'certification time order')
        if self.pin['domain'] == 'synthetic':
            within(original['path'], self.pin['fixture_root'])
        raw = absolute_path(original['path']).read_bytes()
        require(len(raw) == original['size_bytes'] and hashlib.sha256(raw).hexdigest() == original['sha256'],
                'original byte identity drift')
        if role == 'recovery_quiescence':
            body = strict_json(raw.decode('utf-8-sig'))
            require(set(body) == {'owner', 'host', 'namespace', 'observed_at', 'no_live_descendants', 'basis'}
                and digest(body['owner']) == contract['recovery_owner_sha256']
                and body['host'] == body['owner']['process']['host'] and text(body['namespace'])
                and text(body['basis']) and body['no_live_descendants'] is True
                and utc(body['observed_at']) <= utc(m['verified_at']), 'owner/quiescence original mismatch')
        return envelope['signer_id']

    def _revocation(self, contract, signers, now):
        p, r = self.pin, self.pin['revocation']
        message = verify_signature(read_json(absolute_path(r['path'])), {r['authority']: r['public_key']})
        require(isinstance(message, dict) and set(message) == {'version', 'installation_id', 'authority',
            'source_id', 'scope', 'sequence', 'issued_at', 'expires_at', 'revoked_ids', 'untrusted_signers'},
            'revocation schema')
        require(message['version'] == '2.0.0' and message['installation_id'] == p['installation_id']
                and message['authority'] == r['authority'] and message['source_id'] == r['source_id']
                and message['scope'] == p['scope'], 'revocation identity/scope mismatch')
        require(type(message['sequence']) is int and message['sequence'] >= 0, 'revocation sequence')
        issued, expires = utc(message['issued_at']), utc(message['expires_at'])
        require(issued <= now < expires and (now-issued).total_seconds() <= r['max_age_seconds'],
                'revocation unavailable or stale trusted time')
        for field in ('revoked_ids', 'untrusted_signers'):
            values = message[field]
            require(isinstance(values, list) and all(text(x) for x in values) and len(set(values)) == len(values),
                    'revocation identifiers')
        # Consume authenticated state even when it revokes this grant: otherwise
        # a revoked high sequence could be followed by an older permissive one.
        content = digest(message)
        with self.mutex:
            lock = absolute_path(str(self.state_path) + '.lock')
            # The installation prepares both files. Missing lock/state is not bootstrap.
            fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                absolute_path(str(self.state_path))
                previous = read_json(self.state_path)
                require(digest(previous) == self.expected, 'continuity missing, changed or restart checkpoint mismatch')
                require(set(previous) == {'installation_id', 'scope_sha256', 'sequence', 'content_sha256',
                        'issued_at', 'observed_at', 'revoked_ids', 'untrusted_signers'} and previous['installation_id'] == p['installation_id']
                        and previous['scope_sha256'] == digest(p['scope']), 'continuity schema/scope')
                require(type(previous['sequence']) is int and previous['sequence'] >= -1,
                        'continuity sequence')
                require(now >= utc(previous['observed_at']) and issued >= utc(previous['issued_at']),
                        'trusted time rollback')
                require(message['sequence'] >= previous['sequence'] and
                        (message['sequence'] != previous['sequence'] or content == previous['content_sha256']),
                        'revocation rollback or equivocation')
                for field in ('revoked_ids', 'untrusted_signers'):
                    require(isinstance(previous[field], list) and all(text(x) for x in previous[field])
                            and set(previous[field]) <= set(message[field]), 'known revocation set rollback')
                current = dict(installation_id=p['installation_id'], scope_sha256=digest(p['scope']),
                    sequence=message['sequence'], content_sha256=content, issued_at=message['issued_at'], observed_at=now.isoformat(),
                    revoked_ids=message['revoked_ids'], untrusted_signers=message['untrusted_signers'])
                temp = self.state_path.with_name(self.state_path.name + '.' + uuid.uuid4().hex)
                try:
                    with temp.open('xb') as stream:
                        stream.write(canonical(current) + b'\n'); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temp, self.state_path)
                    directory = os.open(self.state_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    self.expected = digest(current)
                finally:
                    temp.unlink(missing_ok=True)
            finally:
                os.close(fd)
        require(contract['revocation_id'] not in message['revoked_ids'], 'approval revoked')
        require(not (set(signers) | {r['authority']}).intersection(message['untrusted_signers']), 'signer no longer trusted')
        return dict(sequence=message['sequence'], content_sha256=content, source_id=r['source_id'])

    def verify(self, contract, approval, now=None):
        now = now or datetime.now(timezone.utc)
        require(now.tzinfo is not None, 'untrusted naive time')
        verify_file(self.pin["installation_record"])
        self._scope(contract)
        require(now < utc(contract['expires_at']), 'approval expired')
        message = verify_signature(approval, self.pin['approval_signers'])
        require(isinstance(message, dict) and set(message) == {'version', 'domain', 'contract_sha256', 'evidence'}
                and message['version'] == '2.0.0' and message['domain'] == contract['domain']
                and message['contract_sha256'] == digest(contract), 'approval binding mismatch')
        evidence = message['evidence']
        require(isinstance(evidence, dict) and set(evidence) == set(ROLES), 'three certified originals required')
        signers = [approval['signer_id']]
        record_ids = {}
        for role in ROLES:
            signers.append(self._evidence(evidence[role], role, contract, now))
            record = evidence[role]['message']
            old = record_ids.setdefault(record['record_id'], digest(record))
            require(old == digest(record), 'certification record identity collision')
        if contract['recovery_owner_sha256'] is not None:
            signers.append(self._evidence(contract['recovery_quiescence'], 'recovery_quiescence', contract, now))
        else:
            require(contract['recovery_quiescence'] is None, 'unbound quiescence')
        state = self._revocation(contract, signers, now)
        return dict(approval_verified=True, domain=contract['domain'],
            real_execution_authorized=contract['domain'] == 'production',
            contract_sha256=digest(contract), revocation=state, trust_installation_id=self.pin['installation_id'])


def production_context():
    global _PRODUCTION_CONTEXT
    require(PRODUCTION_INSTALLATION is not None, 'independent production trust unavailable')
    if _PRODUCTION_CONTEXT is None:
        _PRODUCTION_CONTEXT = TrustContext(PRODUCTION_INSTALLATION)
    return _PRODUCTION_CONTEXT
