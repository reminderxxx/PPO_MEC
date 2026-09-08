"""G14R20-A candidate validation. No execution authority or dispatch capability.

This module deliberately imports no scientific project modules. The separate,
read-only probe loads validators from the pinned scientific worktree.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

VERSION = '1.0.0'
PHASES = ('formal_cache_policy', 'formal_controller', 'formal_ablation',
          'formal_support', 'formal_scalability', 'formal_statistics',
          'formal_gate', 'complete_without_holdout')
V16 = 'typed_model_cache_formal_20260906_152847_g14c_v16'
ROLES = ('protocol', 'bundle', 'environment', 'binding', 'context',
         'command_matrix', 'candidate', 'selection', 'freeze', 'registry',
         'checkpoint', 'resource', 'committed_output')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_json(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def load(path):
    return strict_json(Path(path).read_text(encoding='utf-8-sig'))


def git(root, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    return subprocess.check_output(['git', '-C', str(root), *args],
                                   env=env, text=True).strip()


def safe_path(value):
    p = Path(value)
    if not p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError('non-canonical absolute path: ' + str(value))
    # Reject symlink components, except interpreter paths (handled separately).
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('symlink path: ' + value)
    return p


def identity(path, role):
    p = safe_path(str(path))
    return dict(path=str(p), sha256=file_hash(p), size_bytes=p.stat().st_size, role=role)


def ledger_anchor(path, kind):
    data = Path(path).read_bytes()
    rows = [strict_json(x) for x in data.decode().splitlines()]
    key = 'current_record_hash' if kind == 'phase' else 'current_ledger_hash'
    return dict(path=str(path), kind=kind, record_count=len(rows), byte_count=len(data),
                prefix_sha256=hashlib.sha256(data).hexdigest(), terminal_hash=rows[-1][key],
                run_identity_fingerprint=rows[0]['run_identity_fingerprint'])


def validate_structure(p):
    required = {'version', 'state', 'execution_authorized', 'holdout_capability',
                'run_id', 'run_root', 'worktree_root', 'execution_commit', 'source_tree',
                'evidence', 'ledgers', 'allowed_phases', 'write_scope', 'origin_evidence',
                'executor', 'approval', 'revocation_rules'}
    if not isinstance(p, dict) or set(p) != required:
        raise ValueError('proposal fields mismatch')
    if p['version'] != VERSION or p['state'] != 'proposal':
        raise ValueError('unsupported proposal state/version')
    if p['execution_authorized'] is not False or p['holdout_capability'] is not False:
        raise ValueError('G14R20-A never grants execution or holdout')
    for k in ('run_root', 'worktree_root'):
        safe_path(p[k])
    if Path(p['run_root']).name != p['run_id'] or not p['run_id']:
        raise ValueError('cross-run root')
    if not re.fullmatch('[0-9a-f]{40}', p['execution_commit']):
        raise ValueError('execution commit must be exact')
    if set(p['source_tree']) != {'git_tree', 'tracked_sources_sha256'}:
        raise ValueError('source identity fields')
    for k, n in [('git_tree', 40), ('tracked_sources_sha256', 64)]:
        if not re.fullmatch('[0-9a-f]{%d}' % n, p['source_tree'][k]):
            raise ValueError('source identity hash')
    if p['allowed_phases'] != list(PHASES):
        raise ValueError('only the eight ordered continuation phases are candidates')
    if p['write_scope'] != {'root': p['run_root'], 'mode': 'approved_append_only_successors',
                            'immutable_prefix': True, 'currently_writable': False}:
        raise ValueError('write scope mismatch')
    if p['executor'] != {'state': 'pending', 'commit': None, 'files': []}:
        raise ValueError('external executor is not implemented in A')
    if p['approval'] != {'state': 'pending', 'source': None,
                          'scope': 'implementation_and_read_only_acceptance_only'}:
        raise ValueError('real independent approval cannot be asserted by a proposal')
    if p['revocation_rules'] != ['expired', 'revoked', 'identity_drift', 'failed_terminal',
                                  'prefix_changed', 'unauthorized_successor']:
        raise ValueError('revocation rules mismatch')
    if set(p['origin_evidence']) != {'launch_approval', 'release_attestation'}:
        raise ValueError('origin evidence roles required')
    for row in p['origin_evidence'].values():
        if row != {'status': 'unavailable', 'path': None, 'sha256': None,
                   'boundary': 'not independently verified'}:
            raise ValueError('A cannot self-certify launch/release approval')
    if not isinstance(p['evidence'], list) or not p['evidence']:
        raise ValueError('evidence required')
    seen = set()
    for row in p['evidence']:
        if set(row) != {'path', 'sha256', 'size_bytes', 'role'}:
            raise ValueError('evidence fields')
        safe_path(row['path'])
        if row['path'] in seen or row['role'] not in ROLES:
            raise ValueError('duplicate path or unknown evidence role')
        seen.add(row['path'])
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('evidence full SHA-256 required')
        if type(row['size_bytes']) is not int or row['size_bytes'] < 0:
            raise ValueError('evidence size')
    if set(x['role'] for x in p['evidence']) != set(ROLES):
        raise ValueError('missing mandatory evidence role')
    if len(p['ledgers']) != 2 or {x['kind'] for x in p['ledgers']} != {'phase', 'cell'}:
        raise ValueError('both ledger anchors required')
    for row in p['ledgers']:
        if set(row) != {'path','kind','record_count','byte_count','prefix_sha256',
                        'terminal_hash','run_identity_fingerprint'}:
            raise ValueError('ledger anchor fields')
        expected = 'phase_state.jsonl' if row['kind'] == 'phase' else 'cell_state.jsonl'
        if row['path'] != str(Path(p['run_root']) / expected):
            raise ValueError('ledger outside exact run')
        for k in ('record_count', 'byte_count'):
            if type(row[k]) is not int or row[k] < 1:
                raise ValueError('empty ledger prefix')
        for k in ('prefix_sha256', 'terminal_hash', 'run_identity_fingerprint'):
            if not re.fullmatch('[0-9a-f]{64}', row[k]):
                raise ValueError('ledger hash')
    return {'status': 'pass', 'scope': 'proposal structure only; no evidence or approval implied'}


def validate_ledger(anchor, allowed=()):
    """Read the frozen prefix and legal successors; never construct a ledger writer."""
    raw = safe_path(anchor['path']).read_bytes()
    prefix = raw[:anchor['byte_count']]
    if len(prefix) != anchor['byte_count'] or hashlib.sha256(prefix).hexdigest() != anchor['prefix_sha256']:
        raise ValueError('ledger frozen prefix changed/truncated')
    if not prefix.endswith(b'\n'):
        raise ValueError('anchor must end at a record boundary')
    rows = [strict_json(x) for x in raw.decode().splitlines()]
    if len(prefix.splitlines()) != anchor['record_count']:
        raise ValueError('anchor count mismatch')
    current, previous = (('current_record_hash', 'previous_record_hash')
                         if anchor['kind'] == 'phase' else ('current_ledger_hash', 'previous_ledger_hash'))
    if rows[anchor['record_count'] - 1][current] != anchor['terminal_hash']:
        raise ValueError('anchor terminal mismatch')
    tip = None
    last_phase = -1
    committed = set()
    terminal = set()
    for i, row in enumerate(rows, 1):
        if type(row.get('sequence_number')) is not int or row.get('sequence_number') != i or row.get(previous) != tip:
            raise ValueError('ledger chain sequence/fork')
        if row.get(current) != digest({k: v for k, v in row.items() if k != current}):
            raise ValueError('ledger hash chain')
        if row.get('run_identity_fingerprint') != anchor['run_identity_fingerprint']:
            raise ValueError('cross-run ledger splice')
        if row.get('status') == 'failed':
            raise ValueError('failed terminal remains non-resumable')
        if i > anchor['record_count']:
            phase = row.get('phase')
            if phase not in allowed or phase not in PHASES:
                raise ValueError('unapproved successor phase')
            rank = PHASES.index(phase)
            if rank < last_phase or (anchor['kind'] == 'phase' and rank > last_phase + 1):
                raise ValueError('phase order rollback/skip')
            if anchor['kind'] == 'phase' and rank > last_phase and last_phase >= 0:
                if PHASES[last_phase] not in terminal:
                    raise ValueError('previous phase not completed')
            last_phase = rank
        if row.get('status') == 'committed':
            if row.get('cell_id') in committed:
                raise ValueError('committed result rerun/replacement')
            committed.add(row.get('cell_id'))
        if anchor['kind'] == 'phase':
            if row.get('phase') in terminal:
                raise ValueError('completed phase rerun')
            if row.get('status') == 'completed':
                terminal.add(row['phase'])
        tip = row[current]
    return {'status': 'pass', 'record_count': len(rows), 'prefix_record_count': anchor['record_count'],
            'tip': tip, 'successor_count': len(rows) - anchor['record_count']}


def check_fixture_approval(proposal, approval, *, trusted_key, fixture_root, now=None):
    """Test-only HMAC trust injected by tests, unavailable through CLI.

    Never returns execution_authorized. Fixture root must be independently held
    by the test and the fixture run ID must carry the synthetic prefix.
    """
    root = safe_path(str(fixture_root))
    run = safe_path(proposal['run_root'])
    if not proposal['run_id'].startswith('synthetic_') or V16 in str(run):
        raise ValueError('test trust cannot authorize a real run')
    if root not in run.parents or 'artifacts' in run.parts:
        raise ValueError('fixture outside test trust root')
    body = {k: v for k, v in approval.items() if k != 'signature'}
    expected = hmac.new(trusted_key, canonical(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(approval.get('signature', ''), expected):
        raise ValueError('untrusted approval signature')
    if set(body) != {'proposal_sha256','run_root','scope','expires_at','revoked'}:
        raise ValueError('approval fields')
    if body['proposal_sha256'] != digest(proposal) or body['run_root'] != str(run):
        raise ValueError('approval subject mismatch')
    if body['scope'] != list(PHASES) or body['revoked'] is not False:
        raise ValueError('approval scope/revocation')
    expiry = datetime.fromisoformat(body['expires_at'])
    if expiry.tzinfo is None or expiry <= (now or datetime.now(timezone.utc)):
        raise ValueError('expired approval')
    return {'fixture_approval_valid': True, 'execution_authorized': False,
            'allowed_successors': list(PHASES)}


def preflight(p, mode='evidence'):
    report = {'version': VERSION, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'mode': mode, 'execution_authorized': False, 'holdout_capability': False,
              'dispatch_count': dict(train=0, dev=0, formal=0, holdout=0), 'checks': [],
              'pending': ['independent_continuation_approval', 'external_executor_G14R20_B'],
              'uncovered': ['actual phase dispatch and publication compatibility',
                            'user authorization authenticity', 'formal and holdout evaluation']}
    def check(name, fn, evidence=None):
        try:
            result = fn()
            row = {'name': name, 'status': 'pass', 'detail': result}
        except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            row = {'name': name, 'status': 'fail', 'detail': str(exc)}
        if evidence is not None:
            row['evidence'] = evidence
        report['checks'].append(row)
        return row['status'] == 'pass'
    if not check('structure', lambda: validate_structure(p)):
        return report
    report['proposal_sha256'] = digest(p)
    if mode == 'structure':
        return report
    for row in p['evidence']:
        def verify(row=row):
            actual = identity(row['path'], row['role'])
            if actual != row:
                raise ValueError('immutable evidence drift')
            return actual
        check('file:' + row['role'], verify, row)
    def source():
        root = p['worktree_root']
        if git(root, 'rev-parse', 'HEAD') != p['execution_commit']:
            raise ValueError('execution commit drift')
        if git(root, 'status', '--porcelain'):
            raise ValueError('scientific worktree is dirty')
        if git(root, 'rev-parse', 'HEAD^{tree}') != p['source_tree']['git_tree']:
            raise ValueError('tracked tree drift')
        return {'commit': p['execution_commit'], 'git_tree': p['source_tree']['git_tree']}
    check('clean_fixed_source', source)
    for a in p['ledgers']:
        # A has no production trust store; real successors cannot be approved here.
        check('ledger:' + a['kind'], lambda a=a: validate_ledger(a), a)
    for name, row in p['origin_evidence'].items():
        report['checks'].append({'name': name, 'status': 'unavailable', 'evidence': row})
    report['checks'].extend([
        {'name': 'external_executor', 'status': 'unavailable', 'detail': 'pending G14R20-B'},
        {'name': 'execution_authorization', 'status': 'unavailable', 'detail': 'no independent approval; denied'},
    ])
    return report


def validate_protected_inventory(path, *, immutable_roots=()):
    """Verify every protected byte and detect additions in immutable run/worktree."""
    import gzip
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        inventory = strict_json(f.read())
    observed = set()
    byte_count = 0
    for row in inventory['files']:
        p = Path(row['path'])
        observed.add(str(p))
        if 'symlink' in row:
            if not p.is_symlink() or str(p.readlink()) != row['symlink']:
                raise ValueError('protected symlink changed: ' + str(p))
        else:
            if p.is_symlink() or not p.is_file() or p.stat().st_size != row['size_bytes'] or file_hash(p) != row['sha256']:
                raise ValueError('protected file changed: ' + str(p))
            byte_count += row['size_bytes']
    for root in immutable_roots:
        for p in Path(root).rglob('*'):
            if (p.is_file() or p.is_symlink()) and str(p) not in observed:
                raise ValueError('new immutable file: ' + str(p))
    return {'status': 'pass', 'inventory_sha256': file_hash(path),
            'verified_entries': len(observed), 'verified_bytes': byte_count,
            'scope': 'all recorded protected files; additions checked in listed immutable roots',
            'immutable_roots': list(immutable_roots)}


def validate_committed_payloads(run_root, inventory_path):
    """Reconcile original terminals/markers against the freshly verified inventory.

    The caller must run validate_protected_inventory in this same preflight first.
    This is read-only reconciliation, not a substitute for executor locking.
    """
    import gzip
    with gzip.open(inventory_path, 'rt', encoding='utf-8') as f:
        inventory = {r['path']: r for r in strict_json(f.read())['files']}
    root = safe_path(str(run_root))
    count = 0
    cells = 0
    for record in [strict_json(x) for x in (root / 'cell_state.jsonl').read_text().splitlines()]:
        if record['status'] != 'committed':
            continue
        dest = safe_path(record['committed_path'])
        if root not in dest.parents:
            raise ValueError('committed destination outside run')
        rows = record['artifact_inventory']
        if digest(rows) != record['artifact_inventory_sha256']:
            raise ValueError('committed inventory canonical mismatch')
        marker = load(dest / 'committed_marker.json')
        for field in ('cell_id','run_identity_fingerprint','command_hash','input_hash',
                      'artifact_inventory_sha256','cell_artifact_publication_contract_version'):
            if marker.get(field) != record[field]:
                raise ValueError('marker mismatch: ' + field)
        if marker.get('committed_destination') != str(dest):
            raise ValueError('marker destination mismatch')
        for row in rows:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('inventory escape')
            actual = inventory.get(str(dest / rel), {})
            if any(actual.get(k) != row[k] for k in ('sha256','size_bytes')):
                raise ValueError('committed result replaced: ' + str(dest / rel))
            count += 1
        cells += 1
    phase_files = 0
    for record in [strict_json(x) for x in (root / 'phase_state.jsonl').read_text().splitlines()]:
        if record['status'] != 'completed':
            continue
        for rel, expected in record.get('output_files', {}).items():
            p = Path(rel)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('phase inventory escape')
            if inventory.get(str(root / p), {}).get('sha256') != expected:
                raise ValueError('phase committed output replaced: ' + str(p))
            phase_files += 1
    return {'status':'pass','committed_cells':cells,'cell_inventory_entries':count,
            'phase_output_entries':phase_files,'scope':'original ledger + marker + freshly hashed files'}
