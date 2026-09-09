"""Synthetic-only Ed25519 producer. No production key or root is produced."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .identity import canonical, digest, file_hash, within, ContinuationError
from .production_trust import ROLES, TrustContext, scope_for


def write(path, value):
    Path(path).write_bytes(canonical(value) + b'\n')


def signed(message, signer, key):
    return dict(message=message, signer_id=signer, signature=key.sign(canonical(message)).hex())


class TestTrustFixture:
    __test__ = False

    def __init__(self, contract, *, now=None):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        self.now = now or datetime.now(timezone.utc)
        if contract['domain'] != 'synthetic' or not Path(contract['run_root']).name.startswith('synthetic_'):
            raise ContinuationError('test trust may only serve synthetic runs')
        self.root = within(contract['fixture_root'], contract['fixture_root'])
        within(contract['run_root'], self.root)
        self.contract = contract
        contract.update(version='2.0.0', scientific_commit='a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d',
            release_identity='synthetic-release-only', trust_installation_id='synthetic-installation')
        self.keys = {k: Ed25519PrivateKey.generate() for k in ('approver', 'reviewer', 'revoker')}
        keys = {k: v.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
                for k, v in self.keys.items()}
        self.coord = within(contract['coordination_root'], self.root)
        self.coord.mkdir(exist_ok=True)
        self.state = self.coord / 'trust_continuity_test_only.json'
        self.initial = dict(installation_id=contract['trust_installation_id'], scope_sha256=digest(scope_for(contract)),
            sequence=-1, content_sha256=None, issued_at=(self.now-timedelta(seconds=10)).isoformat(),
            observed_at=(self.now-timedelta(seconds=10)).isoformat(), revoked_ids=[], untrusted_signers=[])
        write(self.state, self.initial)
        Path(str(self.state)+'.lock').touch(exist_ok=False)
        record = self.root/'trust_installation_record_test_only.json'
        write(record, dict(test_only=True, trust_owner='fixture operator', custody='isolated test process',
            scope=scope_for(contract), public_keys=keys))
        self.pin = dict(version='2.0.0', domain='synthetic', fixture_root=str(self.root),
            installation_id=contract['trust_installation_id'], trust_owner='fixture operator',
            installation_record=dict(path=str(record), sha256=file_hash(record), size_bytes=record.stat().st_size),
            scope=scope_for(contract), approval_signers={'approver': keys['approver']},
            verifiers={'reviewer':dict(public_key=keys['reviewer'], author='synthetic author', roles=list(ROLES)+['recovery_quiescence'])},
            revocation=dict(authority='revoker', public_key=keys['revoker'], source_id='test-only-source',
                path=str(self.root/'revocation_test_only.json'), max_age_seconds=7200),
            continuity_path=str(self.state), startup_receipt_path=str(self.coord/'startup_receipt_test_only.json'))
        self.publish()
        self.initial_checkpoint = digest(self.initial)
        self.restart(self.initial_checkpoint)
        self.original = self.root/'original_test_only.json'
        write(self.original, dict(test_only=True, author='synthetic author', responsibilities=list(ROLES),
            run_id=contract['run_id'], scientific_commit=contract['scientific_commit']))
        self.evidence = {role: self.certify(role, self.original) for role in ROLES}
        self.approval = self.approve()
        # Public keys and provenance only. Private fixture keys stay in memory.
        write(self.root/'trust_installation_test_only.json', self.pin)

    def certify(self, role, original):
        scope = scope_for(self.contract)
        if role == 'launch_approval':
            scope['phases'] = ['preflight', 'tests', 'train', 'dev_select', 'checkpoint_freeze']
        if role == 'recovery_quiescence':
            scope['recovery_owner_sha256'] = self.contract['recovery_owner_sha256']
        message = dict(version='2.0.0', record_id='test-record-'+role, verifier_id='reviewer',
            verified_at=self.now.isoformat(), basis='Synthetic fixture author and content checked by test producer; no real qualification',
            original=dict(path=str(original), sha256=file_hash(original), size_bytes=original.stat().st_size,
                author='synthetic author', recorded_at=(self.now-timedelta(seconds=1)).isoformat(), source='isolated fixture producer'),
            scope=scope, roles=[role], coverage={role:'This test original explicitly covers '+role})
        return signed(message, 'reviewer', self.keys['reviewer'])

    def approve(self):
        from .authorization import approval_message
        self.approval = signed(approval_message(self.contract, self.evidence), 'approver', self.keys['approver'])
        return self.approval

    def publish(self, sequence=0, revoked_ids=None, untrusted_signers=None, **changes):
        self.revocation = dict(version='2.0.0', installation_id=self.contract['trust_installation_id'],
            authority='revoker', source_id='test-only-source', scope=scope_for(self.contract), sequence=sequence,
            issued_at=self.now.isoformat(), expires_at=(self.now+timedelta(hours=2)).isoformat(),
            revoked_ids=revoked_ids or [], untrusted_signers=untrusted_signers or [])
        self.revocation.update(changes)
        write(self.pin['revocation']['path'], signed(self.revocation, 'revoker', self.keys['revoker']))

    def verify(self, now=None):
        from .authorization import verify_approval
        return verify_approval(self.contract, self.approval, test_trust_context=self.context, now=now)

    def restart(self, checkpoint=None):
        # Test-side independent custodian explicitly retains the trusted checkpoint.
        # The consumer never derives its startup authority from local state bytes.
        checkpoint = checkpoint or self.context.expected
        self.context = TrustContext(self.pin, test_only=True)
        request = self.context.startup_request()
        message = dict(request, authority='revoker', checkpoint_sha256=checkpoint,
            issued_at=self.now.isoformat(), expires_at=(self.now+timedelta(hours=2)).isoformat())
        write(self.pin['startup_receipt_path'], signed(message, 'revoker', self.keys['revoker']))
        return self.context
