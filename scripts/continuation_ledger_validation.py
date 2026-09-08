"""Read-only prefix, successor, cross-ledger and immutable payload checks for B.

Native validators are supplied by the already verified scientific checkout. No
writer is constructed, no terminal is repaired and no original prefix is rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PHASES = ('formal_cache_policy', 'formal_controller', 'formal_ablation',
          'formal_support', 'formal_scalability', 'formal_statistics',
          'formal_gate', 'complete_without_holdout')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _load_rows(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate ledger JSON field')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('non-finite ledger JSON: ' + value)
    if not data or not data.endswith(b'\n'):
        raise ValueError('ledger must end at a complete record boundary')
    return [json.loads(line, object_pairs_hook=pairs, parse_constant=reject)
            for line in data.decode('utf-8').splitlines()]


def anchored_rows(anchor, *, root, native_validator):
    path = Path(anchor['path'])
    root = Path(root)
    if path.name != anchor['kind'] + '_state.jsonl' or path.parent != root:
        raise ValueError('ledger anchor path/kind mismatch')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('ledger symlink')
    raw = path.read_bytes()
    prefix = raw[:anchor['byte_count']]
    if len(prefix) != anchor['byte_count'] or hashlib.sha256(prefix).hexdigest() != anchor['prefix_sha256']:
        raise ValueError('ledger prefix changed or truncated')
    prefix_rows, rows = _load_rows(prefix), _load_rows(raw)
    if len(prefix_rows) != anchor['record_count']:
        raise ValueError('ledger anchor record count')
    current = 'current_record_hash' if anchor['kind'] == 'phase' else 'current_ledger_hash'
    if prefix_rows[-1][current] != anchor['terminal_hash']:
        raise ValueError('ledger anchor terminal hash')
    native_validator(rows)
    if any(row['run_identity_fingerprint'] != anchor['run_identity_fingerprint'] for row in rows):
        raise ValueError('ledger cross-run identity')
    for row in rows:
        if row['status'] in {'failed', 'failed_terminal'}:
            raise ValueError('terminal failure is not recoverable')
    return rows


def validate_successors(anchors, *, root, phase_validator, cell_validator,
                        artifact_inventory, expected_cells_by_phase):
    if len(anchors) != 2 or {row['kind'] for row in anchors} != {'phase', 'cell'}:
        raise ValueError('exactly one phase and one cell anchor required')
    anchors = {row['kind']: row for row in anchors}
    phases = anchored_rows(anchors['phase'], root=root, native_validator=phase_validator)
    cells = anchored_rows(anchors['cell'], root=root, native_validator=cell_validator)
    prefix_phases = phases[:anchors['phase']['record_count']]
    # Freeze is the immutable start; existing earlier phases cannot be rerun.
    if prefix_phases[-1]['phase'] != 'checkpoint_freeze' or prefix_phases[-1]['status'] != 'completed':
        raise ValueError('approved start is not a completed checkpoint freeze')
    terminals = {row['phase']: row for row in phases if row['status'] == 'completed'}
    for phase in ('train', 'dev_select', 'checkpoint_freeze'):
        if phase not in {row['phase'] for row in prefix_phases if row['status'] == 'completed'}:
            raise ValueError('approved prefix missing prerequisite terminal')
    latest_rank = -1
    phase_rows = {}
    for row in phases[anchors['phase']['record_count']:]:
        phase = row['phase']
        if phase not in PHASES:
            raise ValueError('unauthorized successor phase')
        rank = PHASES.index(phase)
        if rank < latest_rank or rank > latest_rank + 1:
            raise ValueError('phase successor order/skip')
        if rank > latest_rank and latest_rank >= 0:
            if phase_rows[PHASES[latest_rank]][-1]['status'] != 'completed':
                raise ValueError('phase successor precedes previous terminal')
        phase_rows.setdefault(phase, []).append(row)
        latest_rank = rank
    cell_groups = {}
    rank = -1
    immutable_count = 0
    for index, row in enumerate(cells):
        cell_id, phase = row['cell_id'], row['phase']
        group = cell_groups.setdefault(cell_id, [])
        if group and any(row[key] != group[0][key] for key in (
                'phase', 'coordinates', 'command_hash', 'input_hash', 'committed_path')):
            raise ValueError('cell retry changes frozen identity')
        group.append(row)
        if row['status'] == 'failed_retryable':
            if row['return_code'] != 75 or row['retry_allowed'] is not True:
                raise ValueError('only exit 75 is retryable')
            if sum(item['status'] == 'failed_retryable' for item in group) > 1:
                raise ValueError('cell retry exhausted')
        if index >= anchors['cell']['record_count']:
            if phase not in PHASES[:5] or phase not in phase_rows:
                raise ValueError('cell successor lacks matching phase transaction')
            current_rank = PHASES.index(phase)
            if current_rank < rank:
                raise ValueError('cell phase order rollback')
            rank = current_rank
            if cell_id not in expected_cells_by_phase.get(phase, set()):
                raise ValueError('cell outside approved phase matrix')
            if phase_rows[phase][0]['status'] != 'running':
                raise ValueError('cell phase lacks running record')
        if row['status'] == 'committed':
            destination = Path(row['committed_path'])
            if Path(root) not in destination.parents or '.staging' in destination.parts:
                raise ValueError('committed payload outside durable run')
            for part in (destination, *destination.parents):
                if part.is_symlink():
                    raise ValueError('committed payload symlink')
            observed = artifact_inventory(destination)
            if observed != row['artifact_inventory'] or _digest(observed) != row['artifact_inventory_sha256']:
                raise ValueError('immutable committed payload drift')
            marker = destination / 'committed_marker.json'
            if marker.is_symlink() or not marker.is_file():
                raise ValueError('committed marker missing')
            payload = json.loads(marker.read_text())
            for key in ('cell_id', 'run_identity_fingerprint', 'command_hash', 'input_hash',
                        'artifact_inventory_sha256', 'cell_artifact_publication_contract_version'):
                if payload.get(key) != row.get(key):
                    raise ValueError('committed marker identity mismatch: ' + key)
            immutable_count += 1
    for phase, rows in phase_rows.items():
        if rows[-1]['status'] in {'completed', 'completion_candidate'} and phase in PHASES[:5]:
            committed = {row['cell_id'] for row in cells if row['phase'] == phase and row['status'] == 'committed'}
            if committed != set(expected_cells_by_phase.get(phase, set())):
                raise ValueError('phase terminal/cell matrix inconsistency')
    for row in phases:
        if row['status'] not in {'completed', 'completion_candidate'}:
            continue
        for relative, expected in row['output_files'].items():
            path = Path(root) / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('phase output path escape')
            if path.is_symlink() or not path.is_file():
                raise ValueError('phase immutable output missing')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError('phase immutable output changed')
    return {'status': 'pass', 'phase_record_count': len(phases), 'cell_record_count': len(cells),
            'immutable_committed_cells': immutable_count, 'prefix_unchanged': True,
            'freeze_terminal_hash': anchors['phase']['terminal_hash'],
            'phase_tip': phases[-1]['current_record_hash'],
            'cell_tip': cells[-1]['current_ledger_hash']}


def validate_preserved_inventory(inventory_path, *, anchors, scientific_root):
    """Verify original protected bytes while admitting anchored ledger suffixes.

    Unlike A's immutable snapshot validator, only the two explicitly anchored
    ledgers may grow. Every other original protected file remains byte-identical.
    New scientific source files are rejected; legitimate new run outputs are
    checked separately by native publication and successor reconciliation.
    """
    import gzip
    with gzip.open(inventory_path,'rt',encoding='utf-8') as stream:
        inventory=json.load(stream,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite inventory')))
    ledger_anchors={row['path']:row for row in anchors}
    observed=set();verified_bytes=0
    for row in inventory['files']:
        path=Path(row['path'])
        if str(path) in observed:raise ValueError('duplicate protected inventory path')
        observed.add(str(path))
        if str(path) in ledger_anchors:
            anchor=ledger_anchors[str(path)]
            if row.get('sha256')!=anchor['prefix_sha256'] or row.get('size_bytes')!=anchor['byte_count']:
                raise ValueError('inventory and ledger anchor disagree')
            if path.is_symlink():raise ValueError('protected ledger symlink')
            with path.open('rb') as stream:raw=stream.read(anchor['byte_count'])
            if len(raw)!=anchor['byte_count'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('protected ledger prefix drift')
            verified_bytes+=len(raw)
        elif 'symlink' in row:
            if not path.is_symlink() or str(path.readlink())!=row['symlink']:
                raise ValueError('protected symlink drift: '+str(path))
        else:
            if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['size_bytes']:
                raise ValueError('protected file missing/type/size drift: '+str(path))
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=row['sha256']:raise ValueError('protected file content drift: '+str(path))
            verified_bytes+=row['size_bytes']
    if not set(ledger_anchors)<=observed:raise ValueError('protected inventory omits anchored ledger')
    for path in Path(scientific_root).rglob('*'):
        if (path.is_file() or path.is_symlink()) and str(path) not in observed:
            raise ValueError('new scientific worktree file: '+str(path))
    return {'status':'pass','verified_entries':len(observed),'verified_original_bytes':verified_bytes,
        'allowed_append_paths':sorted(ledger_anchors),'scientific_additions':'none'}
