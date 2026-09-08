"""External continuation admission and single-writer primitives (version 1.0.0).

No scientific modules are imported here. Production trust is provisioned separately;
this module never creates approval evidence or keys. Fixture trust is not production
trust, even if its signature is valid.
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Iterator

VERSION = '1.0.0'
PHASES = ('formal_cache_policy', 'formal_controller', 'formal_ablation',
          'formal_support', 'formal_scalability', 'formal_statistics',
          'formal_gate', 'complete_without_holdout')


class AdmissionError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise AdmissionError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def reject(value):
        raise AdmissionError('non-finite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=pairs, parse_constant=reject)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def contained(path, root, *, must_exist=False):
    """Reject lexical escape and every symlink component, including dangling links."""
    root, path = Path(root), Path(path)
    if not root.is_absolute() or not path.is_absolute():
        raise AdmissionError('absolute paths required')
    if '..' in path.parts or '..' in root.parts:
        raise AdmissionError('parent path traversal')
    for candidate in (root, path):
        for part in (candidate, *candidate.parents):
            if part.is_symlink():
                raise AdmissionError('symlink component: ' + str(part))
    if path != root and root not in path.parents:
        raise AdmissionError('path outside approved root: ' + str(path))
    if must_exist and not path.exists():
        raise AdmissionError('missing path: ' + str(path))
    return path


def verify_executor_identity(identity, root):
    if set(identity) != {'version', 'implementation_commit', 'files'} or identity['version'] != VERSION:
        raise AdmissionError('executor identity schema')
    commit = identity['implementation_commit']
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise AdmissionError('executor commit must be full SHA')
    if git(root, 'rev-parse', 'HEAD') != commit:
        raise AdmissionError('executor commit drift')
    if git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise AdmissionError('executor tracked source dirty')
    rows = identity['files']
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise AdmissionError('empty or duplicate executor file inventory')
    required_paths = {str(path.relative_to(root))
                      for path in (Path(root) / 'scripts').glob('*continuation*.py')}
    required_paths.add('src/runtime/fixed_commit_continuation.py')
    if not required_paths <= {row['path'] for row in rows}:
        raise AdmissionError('identity omits continuation implementation/dependency files')
    for row in rows:
        if set(row) != {'path', 'sha256'} or Path(row['path']).is_absolute():
            raise AdmissionError('executor file schema')
        path = contained(Path(root) / row['path'], root, must_exist=True)
        blob = subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + row['path']])
        if file_hash(path) != row['sha256'] or hashlib.sha256(blob).hexdigest() != row['sha256']:
            raise AdmissionError('executor file drift: ' + row['path'])
    # A caller cannot omit the code implementing the check itself.
    if not any((Path(root) / row['path']).resolve() == Path(__file__).resolve() for row in rows):
        raise AdmissionError('identity omits loaded security module')
    return {'fixed_execution_commit': commit,
            'current_branch_observation': git(root, 'rev-parse', '--abbrev-ref', 'HEAD'),
            'files_checked': len(rows)}


def approval_body(contract, *, approval_id, issued_at, expires_at, domain):
    """Build request fields only; this is not signing or approval issuance."""
    return {'version': VERSION, 'domain': domain, 'approval_id': approval_id,
            'contract_sha256': digest(contract),
            'executor_identity_sha256': contract['executor_identity_sha256'],
            'proposal_sha256': contract['proposal_sha256'],
            'run_id': contract['run_id'], 'run_root': contract['run_root'],
            'phases': contract['phases'], 'write_scope': contract['write_scope'],
            'issued_at': issued_at, 'expires_at': expires_at,
            'holdout_capability': False}


def verify_approval(contract, approval, *, phase, domain, verify_signature: Callable,
                    revoked_ids, now=None):
    """Signature verifier and revocation snapshot must come from independent trust."""
    if phase not in PHASES or phase not in contract['phases']:
        raise AdmissionError('unauthorized phase')
    if contract['phases'] != list(PHASES):
        raise AdmissionError('continuation phase order differs')
    if approval is None:
        raise AdmissionError('independent continuation approval unavailable')
    if set(approval) != {'body', 'signature', 'key_id'}:
        raise AdmissionError('approval envelope schema')
    body = approval['body']
    required = {'version', 'domain', 'approval_id', 'contract_sha256',
                'executor_identity_sha256', 'proposal_sha256', 'run_id', 'run_root',
                'phases', 'write_scope', 'issued_at', 'expires_at', 'holdout_capability'}
    if set(body) != required or body['domain'] != domain:
        raise AdmissionError('approval trust domain/schema mismatch')
    expected = approval_body(contract, approval_id=body['approval_id'],
                            issued_at=body['issued_at'], expires_at=body['expires_at'], domain=domain)
    if canonical(body) != canonical(expected):
        raise AdmissionError('approval scope or identity drift')
    if not isinstance(body['approval_id'], str) or not body['approval_id']:
        raise AdmissionError('missing approval identity')
    for key in ('issued_at', 'expires_at'):
        if type(body[key]) not in (int, float):
            raise AdmissionError('invalid approval time')
    now = time.time() if now is None else now
    if not body['issued_at'] <= now < body['expires_at']:
        raise AdmissionError('approval expired or not yet valid')
    if body['approval_id'] in revoked_ids:
        raise AdmissionError('approval revoked')
    if verify_signature(approval['key_id'], canonical(body), approval['signature']) is not True:
        raise AdmissionError('approval signature not trusted')
    return {'approval_id': body['approval_id'], 'domain': domain, 'phase': phase,
            'transaction_admitted_at': now}


class ProductionTrust:
    """Read-only Ed25519 trust interface. No keys are bundled or auto-installed.

    The deployment owner supplies this object from independently reviewed material;
    CLI callers must not choose a key file, trust root or revocation source.
    """
    def __init__(self, public_keys, revoked_ids, signature_backend=None):
        self.signature_backend = signature_backend
        self.public_keys = dict(public_keys)
        self.revoked_ids = frozenset(revoked_ids)

    def verify(self, key_id, message, signature):
        if key_id not in self.public_keys:
            return False
        try:
            spec = importlib.util.spec_from_file_location('external_signature_backend',
                Path(__file__).with_name('continuation_signature_backend.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module.verify(self.signature_backend,self.public_keys[key_id],message,signature)
        except (ValueError, TypeError):
            return False
        except Exception as exc:
            # InvalidSignature and an unavailable crypto backend both fail closed.
            raise AdmissionError('production signature verification failed') from exc


def validate_fixture_root(root):
    spec = importlib.util.spec_from_file_location('external_fixture_sandbox',
        Path(__file__).with_name('continuation_fixture_sandbox.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.validate_fixture_root(root)
    except ValueError as exc:
        raise AdmissionError(str(exc)) from exc


class FixtureTrust:
    """Test-only domain, scoped to a separately created synthetic fixture tree."""
    def __init__(self, root, key, executor_identity_sha256):
        self.root = contained(validate_fixture_root(root), root, must_exist=True)
        if not self.root.name.startswith('synthetic_') or len(key) < 32:
            raise AdmissionError('fixture root/key contract')
        self.key, self.executor_identity_sha256 = key, executor_identity_sha256

    def verify(self, key_id, message, signature):
        return key_id == 'test-only' and isinstance(signature, str) and hmac.compare_digest(
            hmac.new(self.key, message, hashlib.sha256).hexdigest(), signature)

    def validate_contract(self, contract):
        if not contract['run_id'].startswith('synthetic_'):
            raise AdmissionError('fixture cannot reference a real run')
        contained(contract['run_root'], self.root, must_exist=True)
        if Path(contract['run_root']).name != contract['run_id']:
            raise AdmissionError('fixture cross-run identity')
        if contract['executor_identity_sha256'] != self.executor_identity_sha256:
            raise AdmissionError('fixture executor drift')
        for path in contract['write_scope']:
            contained(path, contract['run_root'])

    def validate_references(self, value):
        """Recursively inspect loaded JSON plus nested argv before dispatch.

        Scientific code/interpreter references are checked separately against exact
        approved source inventory, and must not be passed to this input checker.
        """
        if isinstance(value, dict):
            for key, item in value.items():
                self.validate_references(key)
                self.validate_references(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.validate_references(item)
        elif isinstance(value, str):
            if value.startswith('file:') or '://' in value or '..' in Path(value).parts:
                raise AdmissionError('fixture external reference')
            if value.startswith('/'):
                contained(value, self.root)
            if '=' in value:
                self.validate_references(value.split('=', 1)[1])


@contextlib.contextmanager
def single_writer(run_root, *, admission: Callable, owner_identity) -> Iterator[dict]:
    """Authorize before lock creation; kernel lock lifetime follows open file.

    Never unlink/recreate the inode: crash releases flock, an expired token does not
    allow takeover while the original writer is alive. Revalidate after acquisition.
    Owner metadata is audit only, never PID/time-based authority for recovery.
    """
    admission()
    root = contained(run_root, run_root, must_exist=True)
    lock_path = contained(root / '.continuation.lock', root)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            raise AdmissionError('continuation writer already active') from exc
        admitted = admission()
        state = {'version': VERSION, 'owner': owner_identity, 'pid_observation': os.getpid(),
                 'acquired_monotonic_ns': time.monotonic_ns(), 'admission': admitted}
        # Stable bytes are part of the native integrity inventory. Process owner
        # observations live only in the receipt, never rewrite a committed file.
        encoded = canonical({'version': VERSION, 'run_root': str(root),
                             'executor_identity': owner_identity}) + b'\n'
        existing = os.read(fd, len(encoded) + 1)
        if existing and existing != encoded:
            raise AdmissionError('lock fixed executor identity drift')
        if not existing:
            os.write(fd, encoded)
            os.fsync(fd)
        yield state
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def validate_execution_contract(contract):
    required = {'version', 'kind', 'proposal_sha256', 'proposal_file_sha256',
                'executor_identity_sha256', 'run_id', 'run_root', 'scientific_worktree',
                'scientific_commit', 'phases', 'write_scope', 'ledger_anchors',
                'immutable_payload_inventory_sha256', 'origin_evidence',
                'holdout_capability', 'approval_domain'}
    if set(contract) != required or contract['version'] != VERSION:
        raise AdmissionError('execution contract schema')
    if contract['kind'] != 'fixed_commit_continuation_execution_contract':
        raise AdmissionError('execution contract kind')
    if contract['holdout_capability'] is not False or contract['phases'] != list(PHASES):
        raise AdmissionError('execution contract capabilities')
    for key in ('proposal_sha256', 'proposal_file_sha256', 'executor_identity_sha256',
                'immutable_payload_inventory_sha256'):
        if not isinstance(contract[key], str) or not re.fullmatch('[0-9a-f]{64}', contract[key]):
            raise AdmissionError('execution contract hash: ' + key)
    if not re.fullmatch('[0-9a-f]{40}', contract['scientific_commit']):
        raise AdmissionError('execution contract scientific commit')
    root = contained(contract['run_root'], contract['run_root'], must_exist=True)
    if root.name != contract['run_id']:
        raise AdmissionError('execution contract run identity')
    contained(contract['scientific_worktree'], contract['scientific_worktree'], must_exist=True)
    if contract['write_scope'] != [str(root)]:
        raise AdmissionError('execution write scope must be exact existing run')
    if contract['approval_domain'] not in ('production', 'synthetic-only'):
        raise AdmissionError('execution trust domain')
    if contract['approval_domain'] == 'synthetic-only' and not root.name.startswith('synthetic_'):
        raise AdmissionError('synthetic contract cannot name real run')
    if set(contract['origin_evidence']) != {'launch_approval', 'release_attestation'}:
        raise AdmissionError('origin evidence states missing')
    if not isinstance(contract['ledger_anchors'], list) or len(contract['ledger_anchors']) != 2:
        raise AdmissionError('phase and cell ledger anchors required')
    return contract


# Independent deployment boundary, never created or modified by this package.
PRODUCTION_TRUST_PATH = Path('/Library/Application Support/PPO_MEC/continuation_trust.json')


def load_production_trust():
    """Load an independently provisioned, root-owned local trust configuration.

    No CLI/env override, network service, bundled key or first-use enrollment. The
    trusted operator must install the file outside this task. File hash is reported
    with any future verification; its contents are not evidence of old launch or
    release approval, whose separate signed attestations are still required.
    """
    import stat
    path = PRODUCTION_TRUST_PATH
    if not path.is_file():
        raise AdmissionError('independent production trust unavailable; no execution license')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise AdmissionError('production trust path symlink')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        observed = os.fstat(fd)
        if observed.st_uid != 0 or observed.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise AdmissionError('production trust must be independently root-owned and not group/world writable')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read()
    finally:
        os.close(fd)
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AdmissionError('duplicate production trust key')
            result[key] = value
        return result
    config = json.loads(raw, object_pairs_hook=reject_duplicates,
                        parse_constant=lambda value: (_ for _ in ()).throw(AdmissionError('invalid trust JSON')))
    required = {'version', 'domain', 'keys', 'revoked_approval_ids', 'valid_until', 'signature_backend'}
    if set(config) != required or config['version'] != VERSION or config['domain'] != 'production':
        raise AdmissionError('production trust schema/domain')
    if type(config['valid_until']) not in (int, float) or not time.time() < config['valid_until']:
        raise AdmissionError('production trust snapshot expired')
    keys, roles = {}, {}
    for row in config['keys']:
        if set(row) != {'key_id', 'ed25519_public_key_base64', 'roles'}:
            raise AdmissionError('production trust key schema')
        key_id = row['key_id']
        if not isinstance(key_id, str) or not key_id or key_id in keys or 'test' in key_id.lower() or 'synthetic' in key_id.lower():
            raise AdmissionError('production key identity invalid or test-only')
        if not set(row['roles']) <= {'continuation', 'launch', 'release'} or not row['roles']:
            raise AdmissionError('production trust role invalid')
        keys[key_id] = base64.b64decode(row['ed25519_public_key_base64'], validate=True)
        if len(keys[key_id]) != 32:
            raise AdmissionError('production public key size')
        roles[key_id] = frozenset(row['roles'])
    trust = ProductionTrust(keys, config['revoked_approval_ids'], config['signature_backend'])
    trust.roles = roles
    trust.configuration_sha256 = hashlib.sha256(raw).hexdigest()
    trust.valid_until = config['valid_until']
    return trust


def verify_origin_attestations(contract, trust, *, now=None):
    """Launch, release and continuation are separate evidence states."""
    now = time.time() if now is None else now
    reports = {}
    for field, role in [('launch_approval', 'launch'), ('release_attestation', 'release')]:
        item = contract['origin_evidence'][field]
        if not isinstance(item, dict) or item.get('status') != 'independently_verified':
            raise AdmissionError(field + ' unavailable or pending')
        if set(item) != {'status', 'path', 'sha256'}:
            raise AdmissionError(field + ' evidence reference schema')
        path = Path(item['path'])
        if not path.is_absolute() or path.is_symlink() or file_hash(path) != item['sha256']:
            raise AdmissionError(field + ' evidence identity drift')
        evidence = strict_json(path)
        if set(evidence) != {'body', 'signature', 'key_id'}:
            raise AdmissionError(field + ' envelope schema')
        body = evidence['body']
        expected = {'version': VERSION, 'domain': 'production', 'role': role,
                    'run_id': contract['run_id'], 'run_root': contract['run_root'],
                    'scientific_commit': contract['scientific_commit'],
                    'kind': 'original_evidence_verification_record',
                    'original_evidence_path': body.get('original_evidence_path'),
                    'original_evidence_sha256': body.get('original_evidence_sha256'),
                    'evidence_id': body.get('evidence_id'), 'issued_at': body.get('issued_at')}
        if canonical(body) != canonical(expected) or not body['evidence_id']:
            raise AdmissionError(field + ' subject/schema mismatch')
        if type(body['issued_at']) not in (int, float) or body['issued_at'] > now:
            raise AdmissionError(field + ' issued time invalid')
        original = Path(body['original_evidence_path'])
        if not original.is_absolute() or original.is_symlink() or file_hash(original) != body['original_evidence_sha256']:
            raise AdmissionError(field + ' original evidence missing or changed')
        if body['evidence_id'] in trust.revoked_ids:
            raise AdmissionError(field + ' revoked')
        if role not in trust.roles.get(evidence['key_id'], ()):
            raise AdmissionError(field + ' signer not trusted for role')
        if trust.verify(evidence['key_id'], canonical(body), evidence['signature']) is not True:
            raise AdmissionError(field + ' signature invalid')
        reports[field] = {'status': 'independently_verified', 'evidence_sha256': item['sha256'],
                          'key_id': evidence['key_id']}
    return reports


def verify_production_admission(contract, approval, *, phase):
    """Fresh independent trust and all three distinct evidence roles per admission."""
    if contract['approval_domain'] != 'production':
        raise AdmissionError('synthetic approval cannot enter production admission')
    trust = load_production_trust()
    if 'continuation' not in trust.roles.get(approval.get('key_id'), ()):
        raise AdmissionError('signer not trusted for continuation role')
    admitted = verify_approval(contract, approval, phase=phase, domain='production',
        verify_signature=trust.verify, revoked_ids=trust.revoked_ids)
    origins = verify_origin_attestations(contract, trust)
    return {**admitted, 'origin_evidence': origins,
            'trust_configuration_sha256': trust.configuration_sha256,
            'trust_valid_until': trust.valid_until,'signature_backend':trust.signature_backend}
