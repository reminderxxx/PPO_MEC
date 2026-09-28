"""Recompute development evidence without loading models or executing environments."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    def reject(value):
        raise ValueError('non-finite JSON: ' + value)
    return json.loads(path.read_text(), parse_constant=reject)


def episode_record(payload):
    info = payload['run_info']
    metric = payload['formal_request_execution_audit']
    events = [e for e in payload['cache_event_trace'] if e['event_type'] == 'request']
    count = metric['external_request_denominator']
    if not isinstance(count, int) or count <= 0 or len(events) != count:
        raise ValueError('request count mismatch')
    if metric['request_alignment_status'] != 'pass':
        raise ValueError('unaligned requests')
    done = metric['workflow_completed_under_exogenous_execution']
    if type(done) is not bool or any(type(e['service_success']) is not bool for e in events):
        raise ValueError('invalid outcome type')
    transfer = float(metric['transfer_mb_per_request']) * count
    if not math.isfinite(transfer) or transfer < 0:
        raise ValueError('invalid transfer')
    return dict(condition=info['condition_id'], seed=info['seed'], unit=info['unit_id'],
                window=info['window_id'], completed=int(done), requests=count,
                successful_requests=sum(e['service_success'] for e in events),
                total_transfer_mb=transfer)


def aggregate(rows):
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        key = (row['condition'], row['seed'], row['unit'])
        if key in identities:
            raise ValueError('duplicate evaluation identity')
        identities.add(key)
        groups[row['condition']].append(row)
    result = {}
    for condition, group in sorted(groups.items()):
        totals = {key: sum(r[key] for r in group) for key in
                  ('completed', 'requests', 'successful_requests', 'total_transfer_mb')}
        totals['episodes'] = len(group)
        totals['completion_rate'] = totals['completed'] / len(group)
        totals['transfer_per_completed_workflow_including_failure_cost'] = (
            totals['total_transfer_mb'] / totals['completed'] if totals['completed'] else None)
        totals['transfer_per_successful_request_including_failure_cost'] = (
            totals['total_transfer_mb'] / totals['successful_requests']
            if totals['successful_requests'] else None)
        totals['completed_by_seed'] = {
            str(seed): sum(r['completed'] for r in group if r['seed'] == seed)
            for seed in sorted({r['seed'] for r in group})}
        result[condition] = totals
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if args.output_path.resolve().is_relative_to(source):
        raise ValueError('output must be outside immutable source')
    manifest_path = source / 'artifact_integrity_manifest.json'
    manifest = load(manifest_path)
    entries = {}
    for item in manifest['files']:
        rel = item['path']
        path = source / rel
        if rel in entries or path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('unsafe or duplicate source entry')
        if path.stat().st_size != item['size_bytes'] or digest(path) != item['sha256']:
            raise ValueError('source integrity mismatch: ' + rel)
        entries[rel] = item['sha256']
    paths = sorted((source / 'evaluation/episodes').rglob('*.json'))
    rows = []
    for path in paths:
        if path.relative_to(source).as_posix() not in entries:
            raise ValueError('unmanifested episode')
        rows.append(episode_record(load(path)))
    if len(rows) != 156:
        raise ValueError('unexpected frozen matrix size')
    report = dict(
        schema='research_problem_evidence_v1', reviewed_at='2026-09-28',
        source_root=str(source), source_manifest_sha256=digest(manifest_path),
        verified_source_files=len(entries), raw_episode_count=len(rows),
        outer_window_count=len({r['window'] for r in rows}),
        evidence_scope='retrospective_observed_data_development_not_confirmatory',
        estimation_warning='ratios include all failed-episode costs; not causal cost efficiency',
        new_training=0, new_rollout=0, holdout_access=0,
        conditions=aggregate(rows))
    # Recheck all raw inputs after deriving the report, not merely their metadata.
    for path in paths:
        if digest(path) != entries[path.relative_to(source).as_posix()]:
            raise ValueError('episode changed during audit')
    with args.output_path.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
