"""Independently consume one new initial-policy evidence package, without replay.

This audits new physical receipts and references the immutable parent audit.
It never re-runs a parent evaluation or interprets seeds as independent windows.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.complete_stream_initial_evidence import (CONTRACT, SEEDS, CASES, PROBES, REPLAYS,
    EVENT_KEYS, METRICS, FRAME_SUM_METRICS, canonical_hash, validate_plan, load_bindings,
    budget_for, selected_cases, schedule_identity, save_bounded)
from scripts.run_stream_reward_curve import file_hash, json_text, utc_now


def close_number(actual, expected, *, label='native metric'):
    if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
            or not isinstance(expected, (int, float)) or isinstance(expected, bool)
            or not math.isfinite(actual) or not math.isfinite(expected)
            or abs(actual - expected) > 1e-6):
        raise ValueError(label + ' differs or is unavailable')


def episode_key(row):
    return row['phase'], row['seed'], row['case_id']


def expected_matrix():
    return ([('initial', s, c) for s in SEEDS for c in CASES if c not in PROBES]
            + [('replay', r['seed'], r['case_id']) for r in REPLAYS])


def verify_checkpoint(record):
    """Static checkpoint read only: authenticate identity, options and all tensors."""
    if file_hash(record['path']) != record['sha256']:
        raise ValueError('referenced initial checkpoint hash differs')
    payload = torch.load(record['path'], map_location='cpu', weights_only=False)
    identity = {k: v for k, v in record.items() if k not in ('path', 'sha256', 'tensor_sha256')}
    tensors = payload['state_dict']
    digest = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in tensors.values())).hexdigest()
    if (payload.get('reward_curve_identity') != identity or payload.get('options') != record['options']
            or payload.get('contract') != record['checkpoint_contract'] or digest != record['tensor_sha256']
            or int(tensors['completed_updates']) != 0
            or any(payload.get(name, {}).get('state', {}) for name in ('optimizer', 'surrogate_optimizer'))):
        raise ValueError('initial checkpoint class/options/tensors/optimizer identity differs')


def reconstruct(rows, frames_path, checkpoints, cases, data, *, budget=None):
    """Validate every new frame and terminal; never use producer digests as truth."""
    expected = expected_matrix()
    if [episode_key(r) for r in rows] != expected or len(set(expected)) != 48:
        raise ValueError('45 initial plus three fixed replay matrix missing, duplicated or reordered')
    records = {episode_key(r): r for r in rows}
    cp_by = {r['seed']: r for r in checkpoints}
    states = {k: dict(count=0, reward=0., frame_hash=hashlib.sha256(),
        control_hash=hashlib.sha256(), literal_hash=hashlib.sha256(),
        sums=Counter(), status_counts=Counter(), completed={}, event_count=0,
        handoff=0, association=0, target_changes=0, terminal_seen=False) for k in expected}
    sequence = []; prior = None
    with gzip.open(frames_path, 'rt') as stream:
        for line in stream:
            if budget is not None:
                budget.check()
            frame = json.loads(line)
            k = episode_key(frame)
            if k not in records or frame.get('record_kind') != 'native_frame' or frame.get('method') != 'main':
                raise ValueError('undeclared physical frame or method')
            if prior != k:
                sequence.append(k); prior = k
            s, record = states[k], records[k]
            f = frame['frame_index']
            if (type(f) is not int or f != s['count'] or f >= 480 or s['terminal_seen']
                    or frame['frame_after_step'] != f + 1
                    or frame['checkpoint_episode'] != 0
                    or frame['checkpoint_sha256'] != cp_by[k[1]]['sha256']):
                raise ValueError('frame gap/identity/post-terminal interaction')
            if (type(frame['terminal_after_step']) is not bool
                    or type(frame['done_returned']) is not bool
                    or frame['terminal_after_step'] != frame['done_returned']):
                raise ValueError('native terminal flags differ')
            if frame['receipt']['frame_index'] != f or frame['receipt']['reward_components'] != frame['reward_components']:
                raise ValueError('native receipt identity/components differ')
            close_number(frame['reward'], sum(frame['reward_components'].values()), label='frame reward')
            if 'request_rewards' in frame['receipt']:
                close_number(sum(frame['receipt']['request_rewards'].values())
                             + frame['receipt']['unassigned_team_reward'], frame['reward'], label='team receipt shares')
            s['count'] += 1; s['reward'] += frame['reward']
            for name in FRAME_SUM_METRICS:
                value = frame['receipt'].get(name)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('missing/invalid native physical cost: ' + name)
                s['sums'][name] += value
            events, native = frame['events'], frame['native_events']
            if len(events) != len(native):
                raise ValueError('literal native event inventory differs')
            seen = set()
            for event, original in zip(events, native):
                identity = event['vehicle_id'], event['request_id']
                if identity in seen or event['frame_index'] != f:
                    raise ValueError('duplicate/out-of-frame native request event')
                seen.add(identity)
                without_planning = dict(event); without_planning.pop('control_planning', None)
                if without_planning != original or event['vehicle_id'] not in frame['commands']:
                    raise ValueError('native literal event/command correspondence differs')
                if 'control' not in event or not isinstance(event['control_planning'], dict):
                    raise ValueError('missing native control/planning witness')
                if event['control'] != frame['commands'][event['vehicle_id']]:
                    raise ValueError('literal command differs from executed native control')
                s['event_count'] += 1
                s['status_counts'][event['status']] += 1
                s['handoff'] += event['handoff_failed']
                s['association'] += event['association_handoff']
                s['target_changes'] += event.get('service_target_changed', False)
                if event['workflow_completed']:
                    if event['request_id'] in s['completed']:
                        raise ValueError('request completion repeated after completion')
                    s['completed'][event['request_id']] = f
                s['control_hash'].update(json_text({name: event[name] for name in EVENT_KEYS if name in event}).encode())
            if set(frame['commands']) != {v for v, _ in seen}:
                raise ValueError('commands omitted or added relative to native events')
            close_number(sum(e['state_transfer_bytes'] for e in events), frame['receipt']['state_transfer_bytes'])
            close_number(sum(e['state_transfer_link_frames'] for e in events), frame['receipt']['state_transfer_link_frames'])
            declared = schedule_identity(cases[k[2]])
            released = sorted(r['request_id'] for r in declared if r['frame'] <= f + 1)
            if frame['released_request_ids'] != released:
                raise ValueError('late/released request ledger differs')
            horizon = len(data['sessions'][cases[k[2]]['session']]['frames'])
            terminal = f + 1 == horizon or len(s['completed']) == len(declared)
            if frame['terminal_after_step'] != terminal:
                raise ValueError('fabricated or missing physical terminal')
            s['terminal_seen'] = terminal
            s['frame_hash'].update(json_text({name: frame[name] for name in
                                             ('frame_index', 'reward', 'reward_components')}).encode())
            literal = {name: value for name, value in frame.items() if name not in
                       ('phase', 'seed', 'case_id', 'method', 'checkpoint_episode', 'checkpoint_sha256', 'record_kind')}
            s['literal_hash'].update(json_text(literal).encode())
    if sequence != expected:
        raise ValueError('physical episode inventory/order differs')
    results = []
    for k in expected:
        row, s = records[k], states[k]
        cp = cp_by[k[1]]
        declared = schedule_identity(cases[k[2]])
        if (row.get('method') != 'main' or row['checkpoint_sha256'] != cp['sha256']
                or row['checkpoint_episode'] != 0 or row['steps'] != s['count']
                or row['frame_receipt_count'] != s['count'] or not 0 < s['count'] <= 480
                or row['physical_terminal'] is not True or row['bootstrap_team_tail'] != 0
                or not s['terminal_seen'] or row['scheduled_requests'] != declared
                or row['released_request_ids'] != sorted(r['request_id'] for r in declared)
                or row['last_scheduled_release_frame'] != max(r['frame'] for r in declared)
                or row['horizon_frames'] != len(data['sessions'][cases[k[2]]['session']]['frames'])
                or row['request_count'] != len(declared) or row['released_requests'] != len(declared)
                or row['completed_requests'] != len(s['completed'])
                or row['decision_rows'] != s['event_count']
                or row['status_counts'] != dict(s['status_counts'])):
            raise ValueError('native summary/complete terminal/case/checkpoint differs')
        if {r['request_id']: r['completion_frame_index'] for r in row['per_request'] if r['completed']} != s['completed']:
            raise ValueError('per-request completion receipt differs')
        if len(row['per_request']) != len(declared) or {r['request_id'] for r in row['per_request']} != {r['request_id'] for r in declared}:
            raise ValueError('per-request completion matrix differs')
        close_number(row['reward'], s['reward'])
        for name in FRAME_SUM_METRICS: close_number(row[name], s['sums'][name], label=name)
        for name, observed in [('handoff_failed_count', s['handoff']),
                               ('association_handoff_count', s['association']),
                               ('service_target_change_count', s['target_changes'])]:
            close_number(row[name], observed, label=name)
        computed = dict(frame_digest=s['frame_hash'].hexdigest(), control_digest=s['control_hash'].hexdigest(),
                        literal_digest=s['literal_hash'].hexdigest())
        if any(row[name] != value for name, value in computed.items()):
            raise ValueError('new episode digest differs from native frames')
        results.append(dict(phase=k[0], seed=k[1], case_id=k[2], steps=s['count'], reward=s['reward'], **computed))
    by = {episode_key(r): r for r in rows}
    for replay in REPLAYS:
        first = by['initial', replay['seed'], replay['case_id']]
        second = by['replay', replay['seed'], replay['case_id']]
        for name in ('steps', 'frame_digest', 'control_digest', 'literal_digest') + METRICS:
            if first[name] != second[name]:
                raise ValueError('independent initial replay differs: ' + name)
    return results


def attribution(rows, references):
    initial = {(r['seed'], r['case_id']): r for r in rows if r['phase'] == 'initial'}
    for ref in references['reused_initial']:
        r = ref['record']; k = r['seed'], r['case_id']
        if k in initial: raise ValueError('a reused initial probe was physically repeated')
        initial[k] = r
    final = {(r['record']['seed'], r['record']['case_id']): r['record'] for r in references['final_reference']}
    gain = {r['record']['case_id']: r['record'] for r in references['gain_reference']}
    expected = {(s, c) for s in SEEDS for c in CASES}
    if set(initial) != expected or set(final) != expected or set(gain) != set(CASES):
        raise ValueError('complete 60 initial/final and 12 rule records required')
    paired = []
    for seed in SEEDS:
        for case in CASES:
            row = dict(seed=seed, case_id=case, initial_source='reused_probe' if case in PROBES else 'new_initial',
                       initial_checkpoint_sha256=initial[seed, case]['checkpoint_sha256'],
                       final_checkpoint_sha256=final[seed, case]['checkpoint_sha256'])
            for metric in METRICS:
                a, b, g = initial[seed, case][metric], final[seed, case][metric], gain[case][metric]
                for v in (a, b, g): close_number(v, v, label=metric)
                for suffix, v in [('initial', a), ('final', b), ('gain', g), ('learning_delta', b-a),
                                  ('initial_minus_gain', a-g), ('final_minus_gain', b-g)]:
                    row[metric+'_'+suffix] = v
            paired.append(row)
    def aggregate(selected):
        return {k: sum(r[k] for r in selected)/len(selected) for k in paired[0]
                if any(k == metric+'_'+suffix for metric in METRICS for suffix in
                       ('initial', 'final', 'gain', 'learning_delta', 'initial_minus_gain', 'final_minus_gain'))}
    summary = dict(status='descriptive_consumed_train_attribution', initial_records=60, final_records=60,
        unique_physical_rule_records=12, rule_records_logically_referenced_not_rerun=True,
        overall=aggregate(paired), per_seed=[dict(seed=s, **aggregate([r for r in paired if r['seed']==s])) for s in SEEDS],
        per_condition=[dict(case_id=c, **aggregate([r for r in paired if r['case_id']==c])) for c in CASES],
        negative_reward_learning_pairs=[dict(seed=r['seed'], case_id=r['case_id'],
            reward_learning_delta=r['reward_learning_delta']) for r in paired if r['reward_learning_delta']<0],
        negative_completion_learning_pairs=[dict(seed=r['seed'], case_id=r['case_id'],
            completed_requests_learning_delta=r['completed_requests_learning_delta']) for r in paired if r['completed_requests_learning_delta']<0],
        inferential_confidence_intervals_computed=False, paper_goal_complete=False,
        limits=['Consumed training times, no independent/generalization inference.',
               'All five seeds and all 12 input conditions retained; fixed ep000 and ep096 only.',
               'Handoff failures are repeated native events, not distinct failed requests.',
               'Within-main initial/final shares class, priors and policy candidate restrictions.'])
    return paired, summary


def consume(plan_path, expected_sha, *, verify=True):
    plan_path = Path(plan_path)
    if file_hash(plan_path) != expected_sha:
        raise ValueError('prospective consumer plan SHA differs')
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, verify_files=verify)
    output, target = Path(plan['output']), Path(plan['audit_output'])
    target.mkdir(exist_ok=False)
    budget = budget_for(output, plan)
    try:
        terminal = json.loads((output / 'complete.json').read_text())
        if ((output/'failure.json').exists() or terminal.get('status') != 'initial_acquisition_complete'
                or terminal.get('contract') != CONTRACT or terminal.get('plan_sha256') != expected_sha
                or terminal.get('source_commit') != plan['source_commit']
                or terminal.get('initial_episodes') != 45 or terminal.get('replay_episodes') != 3
                or terminal.get('training_frames') != 0 or terminal.get('fit_calls') != 0):
            raise ValueError('failed/wrong new acquisition identity')
        required = {'launch_claim.json', 'protocol.json', 'source_references.json', 'episodes.jsonl', 'frames.jsonl.gz'}
        if set(terminal['output_sha256']) != required:
            raise ValueError('new mandatory raw inventory differs')
        for name, sha in terminal['output_sha256'].items():
            if file_hash(output / name) != sha: raise ValueError('new raw file hash differs')
        protocol = json.loads((output/'protocol.json').read_text())
        claim = json.loads((output/'launch_claim.json').read_text())
        if (protocol != dict(plan=plan, plan_sha256=expected_sha)
                or claim['plan_sha256'] != expected_sha or claim['source_commit'] != plan['source_commit']
                or claim['contract'] != CONTRACT or claim['plan_path'] != str(plan_path.resolve())):
            raise ValueError('launch or plan binding differs')
        if (type(claim['started_unix']) not in (int, float) or not math.isfinite(claim['started_unix'])
                or not 0 < claim['started_unix'] <= time.time()):
            raise ValueError('invalid launch clock')
        budget.deadline = claim['started_unix'] + plan['budgets']['maximum_wall_seconds']
        if terminal['elapsed_seconds'] < 0 or terminal['elapsed_seconds'] >= plan['budgets']['maximum_wall_seconds']:
            raise ValueError('terminal acquisition wall limit exceeded')
        budget.check()
        parent, expected_refs = load_bindings(plan, verify_files=verify)
        references = json.loads((output/'source_references.json').read_text())
        if references != expected_refs:
            raise ValueError('old 15/60/12 record or checkpoint/protocol references differ')
        for record in references['initial_checkpoints']: verify_checkpoint(record)
        data = torch.load(parent['protocol']['protocol_pt'], map_location='cpu', weights_only=False)
        cases = selected_cases(data)
        with (output/'episodes.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        reconstructed = reconstruct(rows, output/'frames.jsonl.gz', references['initial_checkpoints'], cases, data, budget=budget)
        physical = sum(r['steps'] for r in reconstructed)
        if physical != terminal['actual_physical_frames'] or physical > plan['budgets']['maximum_total_physical_frames']:
            raise ValueError('actual new physical frame bill differs or exceeded')
        paired, summary = attribution(rows, references)
        save_bounded(target/'summary.json', summary, budget)
        text = io.StringIO(); csv_writer=csv.DictWriter(text, fieldnames=list(paired[0]));csv_writer.writeheader();csv_writer.writerows(paired)
        budget.check(len(text.getvalue().encode()))
        with (target/'attribution.csv').open('x') as stream: stream.write(text.getvalue())
        result = dict(status='initial_evidence_audited', contract=CONTRACT, reviewed_at=utc_now(),
            plan_sha256=expected_sha, source_commit=plan['source_commit'],
            parent_campaign=plan['parent_campaign'], parent_audit=plan['parent_audit'],
            historical_source=references['historical_source'], execution_source=references['execution_source'],
            source_compatibility=references['source_compatibility'],
            new_initial_episodes=45, reused_initial_probes=15, reused_final_episodes=60,
            reused_rule_physical_records=12, new_initial_replays=3, replay_digests_match=True,
            actual_physical_frames=physical, training_frames=0, fit_calls=0,
            source_references_sha256=file_hash(output/'source_references.json'),
            acquisition_complete_sha256=file_hash(output/'complete.json'),
            raw_inventory=terminal['output_sha256'], reconstructed=reconstructed,
            summary_sha256=file_hash(target/'summary.json'), attribution_csv_sha256=file_hash(target/'attribution.csv'),
            paper_goal_complete=False, inferential_statistics=False,
            limits=summary['limits'])
        save_bounded(target/'audit.json', result, budget)
        budget.check()
        complete = dict(status='initial_attribution_evidence_complete', audit=str(target/'audit.json'),
            audit_sha256=file_hash(target/'audit.json'), summary=str(target/'summary.json'),
            paired_csv=str(target/'attribution.csv'), actual_physical_frames=physical,
            consumed_train_only=True, paper_goal_complete=False)
        save_bounded(target/'complete.json', complete, budget)
        return complete
    except BaseException as exc:
        with (target/'failure.json').open('x') as stream:
            stream.write(json_text(dict(status='failed_preserved_readonly_consumer', failed_at=utc_now(),
                error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                plan_sha256=expected_sha, scientific_interactions=0, paper_goal_complete=False)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--expected_sha', required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(consume(args.plan, args.expected_sha)))


if __name__ == '__main__': main()
