"""Read-only completion consumer for the frozen four-method 504-session run.

No model deserialization, environment calls, peak selection or new evaluation.
Matplotlib is imported only by render(); verification/statistics use stdlib.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'stream_long_convergence_report_v1'
METHODS = ['sa_ghmappo', 'mappo', 'ppo', 'dag_offload_drl']
SEEDS = [7, 17, 27]
FULL_POINTS = list(range(0, 505, 72))
PROBE_POINTS = list(range(0, 505, 24))
LABELS = dict(sa_ghmappo='SA-GHMAPPO', mappo='MAPPO', ppo='PPO', dag_offload_drl='DAG-offload DRL')
MAX_LINE = 1024 * 1024
MAX_JSON = 16 * 1024 * 1024
MAX_RSS = 512 * 1024 * 1024
MAX_SECONDS = 3600
SHA_CHARS = set('0123456789abcdef')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def equal(actual, expected, message):
    if isinstance(actual, (set, Path)) and isinstance(expected, type(actual)):
        require(actual == expected, message)
    else:
        require(canonical(actual) == canonical(expected), message)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(wire):
    return json.loads(wire, object_pairs_hook=unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def digest_string(value):
    return type(value) is str and len(value) == 64 and set(value) <= SHA_CHARS


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid integer: ' + label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid number: ' + label)
    return value


class Budget:
    def __init__(self, *, clock=time.monotonic, rss=None, allow_synthetic=False):
        self.clock = clock
        self.started = clock()
        self.rss = rss or self.current_rss
        self.peak_rss = 0
        self.inputs = {}
        self.allow_synthetic = allow_synthetic
        self.synthetic_seen = False

    @staticmethod
    def current_rss():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return int(value)
        if sys.platform.startswith('linux'):
            return int(value) * 1024
        raise RuntimeError('Unsupported RSS platform')

    def check(self):
        self.peak_rss = max(self.peak_rss, self.rss())
        if self.peak_rss > MAX_RSS:
            raise RuntimeError('Read-only consumer RSS limit exceeded')
        if self.clock() - self.started > MAX_SECONDS:
            raise RuntimeError('Read-only consumer wall-time limit exceeded')

    def observed(self, path, snapshot, digest=None):
        name = str(Path(path).absolute())
        previous = self.inputs.get(name)
        if previous is not None:
            equal(snapshot, previous[0], 'Input changed during read: ' + name)
            if digest is not None and previous[1] is not None:
                equal(digest, previous[1], 'Input digest changed during read: ' + name)
        self.inputs[name] = (snapshot, digest if digest is not None else (previous[1] if previous else None))

    def unchanged(self):
        for name, (snapshot, _) in self.inputs.items():
            equal(file_snapshot(regular(name)), snapshot, 'Input changed before completion: ' + name)
            self.check()

    def note_synthetic(self, value):
        stack = [value]
        while stack:
            item = stack.pop()
            if type(item) is dict:
                if item.get('synthetic_fixture') is True or item.get('not_scientific_evidence') is True:
                    self.synthetic_seen = True
                    require(self.allow_synthetic, 'Known synthetic evidence rejected; explicit software-only QA mode required')
                stack.extend(child for child in item.values() if type(child) in (dict, list))
            elif type(item) is list:
                stack.extend(child for child in item if type(child) in (dict, list))
        self.check()


def regular(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'Regular non-symlink file required: ' + str(path))
    return path


def file_snapshot(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def sha_file(path, budget):
    digest = hashlib.sha256()
    path = regular(path)
    before = file_snapshot(path)
    with path.open('rb') as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            budget.check()
    budget.check()
    equal(file_snapshot(path), before, 'Input changed while hashing')
    result = digest.hexdigest()
    budget.observed(path, before, result)
    return result


def read_json(path, budget):
    path = regular(path)
    before = file_snapshot(path)
    require(path.stat().st_size <= MAX_JSON, 'JSON size limit exceeded')
    with path.open('rb') as source:
        wire = source.read(MAX_JSON + 1)
    require(len(wire) <= MAX_JSON, 'JSON grew beyond size limit')
    equal(file_snapshot(path), before, 'JSON changed while reading')
    budget.observed(path, before, hashlib.sha256(wire).hexdigest())
    budget.check()
    value = decode(wire)
    budget.note_synthetic(value)
    return value


def rows(path, budget, count, max_bytes):
    path = regular(path)
    before = file_snapshot(path)
    require(path.stat().st_size <= max_bytes, 'JSONL file byte cap exceeded')
    total = 0
    with path.open('rb') as source:
        while True:
            wire = source.readline(MAX_LINE + 1)
            if not wire:
                break
            require(len(wire) <= MAX_LINE and wire.endswith(b'\n'), 'Oversized or unterminated JSONL line')
            total += 1
            require(total <= count, 'Too many JSONL rows')
            budget.check()
            row = decode(wire)
            require(type(row) is dict, 'JSONL object required')
            budget.note_synthetic(row)
            yield row
    require(total == count, 'Missing JSONL rows')
    equal(file_snapshot(path), before, 'JSONL changed while reading')
    budget.observed(path, before)


def bind(spec, budget):
    require(type(spec) is dict and set(spec) == {'path', 'sha256'}, 'Exact path/SHA binding required')
    require(digest_string(spec['sha256']), 'Invalid bound SHA')
    path = Path(spec['path'])
    require(path.is_absolute(), 'Absolute bound path required')
    equal(sha_file(path, budget), spec['sha256'], 'Bound file SHA differs: ' + str(path))
    return path


def producer():
    spec = importlib.util.spec_from_file_location('plot_stream_long_stdlib_producer', ROOT / 'scripts/train_stream_long.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(row):
    """Project only raw summary fields; do not retain per-request metadata."""
    reward = number(row.get('reward'), 'reward')
    requested = integer(row.get('request_count'), 'request_count', 1)
    completed = integer(row.get('completed_requests'), 'completed_requests')
    require(completed <= requested, 'Completed request count exceeds requests')
    steps = integer(row.get('steps'), 'steps', 1)
    require(steps <= 480, 'Session frame cap exceeded')
    cache = row.get('model_cache')
    require(type(cache) is dict and len(cache) == 4, 'Four-RSU model_cache summary required')
    transferred = 0
    for value in cache.values():
        require(type(value) is dict, 'Invalid model_cache RSU summary')
        transferred += number(value.get('transferred_bytes'), 'transferred_bytes')
        require(value['transferred_bytes'] >= 0, 'Negative transferred bytes')
    return dict(reward=reward, request_count=requested, completed_requests=completed,
        transferred_bytes=transferred, steps=steps)


def aggregate(values):
    require(bool(values), 'Empty aggregation')
    requested = sum(value['request_count'] for value in values)
    completed = sum(value['completed_requests'] for value in values)
    transferred = sum(value['transferred_bytes'] for value in values)
    return dict(reward=statistics.mean(value['reward'] for value in values),
        request_completion_percent=100 * completed / requested,
        model_kib_per_completed_request=transferred / 1024 / completed if completed else None,
        request_count=requested, completed_requests=completed, transferred_bytes=transferred,
        row_count=len(values))


def relative_spread(values):
    if any(value is None for value in values):
        return None
    mean = statistics.mean(values)
    return (max(values) - min(values)) / abs(mean) if mean else None


def tail_check(points, criteria):
    values = [points[episode] for episode in criteria['episodes']]
    reward = relative_spread([value['reward'] for value in values])
    completion = max(value['request_completion_percent'] for value in values) - min(value['request_completion_percent'] for value in values)
    cost = relative_spread([value['model_kib_per_completed_request'] for value in values])
    checks = dict(reward=reward is not None and reward <= criteria['reward_relative_range'],
        completion=completion <= criteria['completion_percentage_point_range'],
        model_cost=cost is not None and cost <= criteria['model_cost_relative_range'])
    return dict(episodes=criteria['episodes'], reward_relative_range=reward,
        completion_percentage_point_range=completion, model_cost_relative_range=cost,
        checks=checks, passed=all(checks.values()))


def checkpoint_identity(value, method, seed, episode, plan, protocol_sha):
    expected = dict(protocol=plan['schema'], protocol_id=plan['protocol_id'], protocol_sha256=protocol_sha,
        source_commit=plan['source_commit'], method=method, seed=seed, episode=episode,
        training_budget=504, completed_updates=episode // 4, resume_supported=False, primary=episode == 504,
        output_byte_count_scope=plan['output_byte_count_scope'], checkpoint_charged_bytes=plan['limits']['checkpoint_bytes'])
    for key, wanted in expected.items():
        equal(value.get(key), wanted, 'Checkpoint identity differs: ' + key)
    require(digest_string(value.get('checkpoint_sha256')), 'Invalid checkpoint SHA')
    actual = integer(value.get('checkpoint_actual_bytes'), 'checkpoint_actual_bytes', 1)
    written = integer(value.get('checkpoint_write_bytes'), 'checkpoint_write_bytes', 1)
    integer(value.get('checkpoint_save_calls'), 'checkpoint_save_calls', 1)
    require(actual <= written <= plan['limits']['checkpoint_bytes'], 'Checkpoint byte accounting differs')


def counts_check(value, expected, limits):
    require(type(value) is dict and set(value) == set(expected), 'Operation counter inventory differs')
    for key, wanted in expected.items():
        integer(value[key], key)
        require(value[key] <= limits[key], 'Operation counter exceeds limit: ' + key)
        if wanted is not None:
            equal(value[key], wanted, 'Operation counter differs: ' + key)


def verify(protocol, protocol_sha256=None, *, budget=None, allow_synthetic=False):
    budget = budget or Budget(allow_synthetic=allow_synthetic)
    equal(budget.allow_synthetic, allow_synthetic, 'Synthetic QA mode must be explicit and consistent')
    budget.check()
    protocol = regular(Path(protocol).absolute())
    protocol_sha256 = protocol_sha256 or sha_file(protocol, budget)
    require(digest_string(protocol_sha256), 'Frozen protocol SHA required')
    equal(sha_file(protocol, budget), protocol_sha256, 'Protocol SHA differs')
    plan = read_json(protocol, budget)
    p = producer()
    # The original decision gate is stdlib only. Route all of its I/O through
    # the same streaming/hash/JSON bounds as this independent consumer.
    p.sha_file = lambda path: sha_file(path, budget)
    p.read_json = lambda path: read_json(path, budget)
    p.validate_plan(plan)
    require(plan['schema'] == 'joint_stream_long_four_v1', 'Four-method protocol required')
    equal(plan['protocol_id'], p.protocol_identity(plan), 'Protocol semantic identity differs')
    equal(plan['methods'], METHODS, 'Four exact methods required')
    equal(plan['seeds'], SEEDS, 'Three exact seeds required')
    equal(protocol.resolve(), (Path(plan['output_root']) / 'manifest.json').resolve(), 'Manifest location differs')
    require(type(plan.get('source_commit')) is str and len(plan['source_commit']) == 40 and set(plan['source_commit']) <= SHA_CHARS, 'Invalid source commit')
    repo = Path(plan['repo_root'])
    expected_inventory = {str(path.relative_to(repo)) for path in (repo / 'src').rglob('*.py')} | {str(path.relative_to(repo)) for path in (repo / 'scripts').glob('*.py')}
    equal(set(plan['source_fingerprints']), expected_inventory, 'Frozen source inventory differs')
    for name, expected in plan['source_fingerprints'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Source path escape')
        equal(sha_file(repo / relative, budget), expected, 'Frozen source SHA differs: ' + name)
    original_path = bind(plan['source_protocol'], budget)
    original = read_json(original_path, budget)
    for key, expected in dict(protocol='joint_stream_learning_v2', action_contract='joint_stream_control_v2', smoke=False, episodes=64, seeds=SEEDS, train_case_count=72, dev_case_count=48).items():
        equal(original.get(key), expected, 'Original protocol differs: ' + key)
    equal(original.get('curve_probe_ids'), plan['probe_case_ids'], 'Original probe identity differs')
    equal(set(original['assets']), {'protocol.pt', 'predictor.pt'}, 'Original asset inventory differs')
    expected_assets = {str((original_path.parent / name).resolve()): expected for name, expected in original['assets'].items()}
    equal(plan['source_assets'], expected_assets, 'Source asset mapping differs')
    for path, expected in expected_assets.items():
        bind(dict(path=path, sha256=expected), budget)
    root = Path(plan['run_root'])
    require(root.is_dir() and not root.is_symlink(), 'Completed run directory required')
    require(not list(root.rglob('failure.json')), 'A failed run cannot produce convergence evidence')
    result_path = root / 'result.json'
    result = read_json(result_path, budget)
    for key, expected in dict(schema=plan['schema'], status='complete_fixed504', protocol_sha256=protocol_sha256,
        protocol_id=plan['protocol_id'], predictor_fit_calls=0, formal_holdout='not_read', paper_claim_ready=False).items():
        equal(result.get(key), expected, 'Root result differs: ' + key)
    equal(result['limits'], plan['limits'], 'Root limits differ')
    equal(result['tail_stability'], dict(criteria=plan['tail_stability'], audit_status='pending_independent_consumer'), 'Root tail criteria differ')
    decision_path = bind(result['decision'], budget)
    decision = read_json(decision_path, budget)
    equal(decision.get('schema'), p.DECISION_SCHEMA, 'Decision schema differs')
    equal(decision.get('protocol_sha256'), protocol_sha256, 'Decision protocol SHA differs')
    # The producer rechecked the full intervention decision before/after training.
    # Re-run its stdlib gate rather than reinterpret diagnostic eligibility here.
    bound_decision, _ = p.check_decision(decision_path, protocol, plan)
    equal(bound_decision, result['decision'], 'Decision binding differs')
    started_path = root / 'run_started.json'
    started = read_json(started_path, budget)
    for key, expected in dict(schema=plan['schema'], protocol_sha256=protocol_sha256, source_commit=plan['source_commit'],
        decision=result['decision'], no_resume=True, no_retry=True).items():
        equal(started.get(key), expected, 'Run command provenance differs: ' + key)
    require(type(started.get('command')) is list and bool(started['command']), 'Run command log required')
    command = started['command']
    require(len(command) == 6 and all(type(value) is str for value in command), 'Exact producer command required')
    equal(Path(command[0]).resolve(), (repo / 'scripts/train_stream_long.py').resolve(), 'Run command entry differs')
    equal(command[1], 'run', 'Run command action differs')
    options = dict(zip(command[2::2], command[3::2]))
    equal(set(options), {'--protocol', '--decision_receipt'}, 'Run command option inventory differs')
    equal(Path(options['--protocol']).resolve(), protocol.resolve(), 'Run command protocol differs')
    equal(Path(options['--decision_receipt']).resolve(), decision_path.resolve(), 'Run command decision differs')
    allowed_root = {'result.json', 'run_started.json', 'progress.json', *METHODS}
    require(set(path.name for path in root.iterdir()) <= allowed_root, 'Unexpected run-root file/directory')
    equal({path.name for path in root.iterdir() if path.is_dir()}, set(METHODS), 'Method directory inventory differs')
    require(type(result.get('cells')) is list and len(result['cells']) == 12, 'Twelve completed cells required')
    root_cells = {}
    for cell in result['cells']:
        key = (cell.get('method'), cell.get('seed'))
        require(key not in root_cells, 'Duplicate root cell')
        root_cells[key] = cell
    equal(set(root_cells), {(method, seed) for method in METHODS for seed in SEEDS}, 'Root cell grid differs')
    dev, train, bindings, cell_counts = {}, {}, [], []
    counter_keys = ('learner_models', 'evaluation_models', 'predictor_models', 'training_sessions',
        'evaluation_sessions', 'environments', 'training_frames', 'evaluation_frames', 'learner_updates',
        'backward_calls', 'main_optimizer_steps', 'surrogate_optimizer_steps', 'categorical_sample_calls',
        'evaluate_calls', 'checkpoints', 'output_bytes')
    known = dict(learner_models=1, evaluation_models=22, predictor_models=1056, training_sessions=504,
        evaluation_sessions=552, environments=1056, learner_updates=126, checkpoints=22)
    expected_names = {'training.jsonl', 'dev.jsonl', 'updates.jsonl', 'rng_isolation.jsonl', 'final_events.jsonl.gz', 'final_receipts.jsonl.gz'}
    expected_names |= {'ep%03d%s' % (episode, suffix) for episode in PROBE_POINTS for suffix in ('.pt', '.identity.json')}
    for method in METHODS:
        require(not (root / method).is_symlink(), 'Symlink method directory forbidden')
        equal({path.name for path in (root / method).iterdir()}, {'seed' + str(seed) for seed in SEEDS}, 'Seed directory inventory differs')
        for seed in SEEDS:
            home = root / method / ('seed' + str(seed))
            require(home.is_dir() and not home.is_symlink(), 'Regular cell directory required')
            receipt_path = home / 'result.json'
            receipt = read_json(receipt_path, budget)
            equal(receipt, root_cells[(method, seed)], 'Cell/root receipts differ')
            for key, expected in dict(schema=plan['schema'], status='complete_fixed504', method=method, seed=seed,
                primary_checkpoint='ep504.pt', predictor_fit_calls=0, formal_holdout='not_read', no_peak_selection=True).items():
                equal(receipt.get(key), expected, 'Cell completion differs: ' + key)
            equal(set(receipt['output_sha256']), expected_names, 'Cell output SHA inventory differs')
            equal({path.name for path in home.iterdir()}, expected_names | {'result.json'}, 'Cell output inventory differs')
            files = {}
            for name in sorted(expected_names):
                path = regular(home / name)
                cap = (plan['limits']['checkpoint_bytes'] if name.endswith('.pt') else MAX_JSON if name.endswith('.identity.json') else
                    plan['limits']['final_events_bytes_per_cell'] if name == 'final_events.jsonl.gz' else
                    plan['limits']['final_receipts_bytes_per_cell'] if name == 'final_receipts.jsonl.gz' else
                    plan['limits']['training_jsonl_bytes_per_cell'] if name == 'training.jsonl' else
                    plan['limits']['dev_jsonl_bytes_per_cell'] if name == 'dev.jsonl' else
                    plan['limits']['updates_jsonl_bytes_per_cell'] if name == 'updates.jsonl' else plan['limits']['isolation_jsonl_bytes_per_cell'])
                require(path.stat().st_size <= cap, 'Cell output file byte cap exceeded: ' + name)
                expected = receipt['output_sha256'][name]
                require(digest_string(expected), 'Invalid output SHA')
                equal(sha_file(path, budget), expected, 'Output bytes differ: ' + str(path))
                files[name] = dict(path=str(path), sha256=expected, bytes=path.stat().st_size)
            identities = {}
            require(type(receipt['checkpoints']) is list and len(receipt['checkpoints']) == 22, 'Twenty-two checkpoints required')
            for identity in receipt['checkpoints']:
                episode = identity.get('episode')
                require(type(episode) is int and episode in PROBE_POINTS and episode not in identities, 'Duplicate/unfrozen checkpoint')
                checkpoint_identity(identity, method, seed, episode, plan, protocol_sha256)
                equal(read_json(home / ('ep%03d.identity.json' % episode), budget), identity, 'Checkpoint identity sidecar differs')
                cp = files['ep%03d.pt' % episode]
                equal(cp['sha256'], identity['checkpoint_sha256'], 'Checkpoint byte SHA differs')
                equal(cp['bytes'], identity['checkpoint_actual_bytes'], 'Checkpoint byte size differs')
                identities[episode] = identity
            seen = set()
            for row in rows(home / 'rng_isolation.jsonl', budget, 22, plan['limits']['isolation_jsonl_bytes_per_cell']):
                episode = row.get('episode')
                require(type(episode) is int and episode in identities and episode not in seen, 'Duplicate/unfrozen isolation point')
                seen.add(episode)
                for key, expected in identities[episode].items():
                    equal(row.get(key), expected, 'Isolation checkpoint identity differs: ' + key)
                for before, after in (('learner_before', 'learner_after'), ('rng_before', 'rng_after')):
                    require(digest_string(row.get(before)) and row[before] == row.get(after), 'Learner/RNG isolation failed')
                equal(row.get('rng_and_learner_unchanged'), True, 'Isolation success missing')
                equal(row.get('unique_evaluation_cases'), 48 if episode in FULL_POINTS else 12, 'Isolation case count differs')
            dev_rows = {}
            for row in rows(home / 'dev.jsonl', budget, 552, plan['limits']['dev_jsonl_bytes_per_cell']):
                episode, case_id = row.get('episode'), row.get('case_id')
                require(type(episode) is int and episode in PROBE_POINTS, 'Unfrozen dev point')
                eligible = plan['dev_case_ids'] if episode in FULL_POINTS else plan['probe_case_ids']
                require(case_id in eligible and (episode, case_id) not in dev_rows, 'Duplicate/unknown dev case')
                for key, expected in dict(protocol_id=plan['protocol_id'], method=method, seed=seed,
                    checkpoint_sha256=identities[episode]['checkpoint_sha256']).items():
                    equal(row.get(key), expected, 'Dev provenance differs: ' + key)
                panels = (['full48'] if episode in FULL_POINTS else []) + (['probe12'] if case_id in plan['probe_case_ids'] else [])
                equal(row.get('panels'), panels, 'Probe rows must be reused at full points; panels differ')
                dev_rows[(episode, case_id)] = metrics(row)
            expected_dev = {(episode, case_id) for episode in PROBE_POINTS for case_id in (plan['dev_case_ids'] if episode in FULL_POINTS else plan['probe_case_ids'])}
            equal(set(dev_rows), expected_dev, 'Incomplete dev panel grid')
            train_rows = {}
            for row in rows(home / 'training.jsonl', budget, 504, plan['limits']['training_jsonl_bytes_per_cell']):
                episode = row.get('episode')
                require(type(episode) is int and 1 <= episode <= 504 and episode not in train_rows, 'Duplicate/unknown training episode')
                for key, expected in dict(method=method, seed=seed, case_id=plan['training_schedule'][str(seed)][episode - 1]).items():
                    equal(row.get(key), expected, 'Training schedule/provenance differs: ' + key)
                train_rows[episode] = dict(case_id=row['case_id'], **metrics(row))
            seen_updates = set()
            for row in rows(home / 'updates.jsonl', budget, 126, plan['limits']['updates_jsonl_bytes_per_cell']):
                episode = row.get('episode')
                require(type(episode) is int and episode in range(4, 505, 4) and episode not in seen_updates, 'Duplicate/unknown update episode')
                seen_updates.add(episode)
                for key, expected in dict(method=method, seed=seed, update=episode // 4).items():
                    equal(row.get(key), expected, 'Update provenance differs: ' + key)
            expected_counts = {key: known.get(key) for key in counter_keys}
            expected_counts['training_frames'] = sum(value['steps'] for value in train_rows.values())
            expected_counts['evaluation_frames'] = sum(value['steps'] for value in dev_rows.values())
            counts_check(receipt['counts'], expected_counts, plan['limits'])
            require(receipt['counts']['main_optimizer_steps'] >= 126 and receipt['counts']['backward_calls'] >= 126 and
                receipt['counts']['categorical_sample_calls'] > 0 and receipt['counts']['evaluate_calls'] > 0,
                'Completed training requires recorded actual learning/sampling calls')
            cell_counts.append(receipt['counts'])
            dev[(method, seed)], train[(method, seed)] = dev_rows, train_rows
            bindings.append(dict(method=method, seed=seed, result=dict(path=str(receipt_path), sha256=sha_file(receipt_path, budget)), files=files))
    expected_totals = {key: (None if key == 'output_bytes' else sum(count[key] for count in cell_counts)) for key in counter_keys}
    counts_check(result['counts'], expected_totals, plan['limits'])
    for key in ('learner_models', 'evaluation_models', 'predictor_models', 'training_sessions', 'evaluation_sessions', 'environments', 'learner_updates', 'checkpoints'):
        equal(result['counts'][key], plan['limits'][key], 'Incomplete frozen total: ' + key)
    require(result['counts']['output_bytes'] >= sum(count['output_bytes'] for count in cell_counts), 'Root output byte charges less than cells')
    report = statistics_report(plan, dev)
    # Recheck raw projections' inputs after streaming, so a concurrent rewrite
    # cannot substitute rows after their initial inventory digest.
    for cell in bindings:
        for name in ('training.jsonl', 'dev.jsonl', 'updates.jsonl', 'rng_isolation.jsonl'):
            equal(sha_file(cell['files'][name]['path'], budget), cell['files'][name]['sha256'], 'Input changed while reading')
    equal(sha_file(protocol, budget), protocol_sha256, 'Protocol changed while reading')
    budget.unchanged()
    equal({str(path.relative_to(repo)) for path in (repo / 'src').rglob('*.py')} | {str(path.relative_to(repo)) for path in (repo / 'scripts').glob('*.py')}, expected_inventory, 'Source inventory changed during read')
    report.update(schema=SCHEMA, status='SYNTHETIC_SOFTWARE_PREVIEW' if allow_synthetic else 'complete_read_only_consumer',
        synthetic_fixture=allow_synthetic, scientific_execution=False,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        literature_cutoff=None, target_venue='Development evidence only; no venue readiness decision',
        artifact_run_id=plan['protocol_id'], policy_version='tmc_review_policy_v3_20260621', git_commit=plan['source_commit'],
        evidence_level='SOFTWARE_SYNTHETIC_NOT_SCIENTIFIC' if allow_synthetic else 'DEVELOPMENT_RAW_ARTIFACT_AUDITED_FORMAL_E0_UNAVAILABLE',
        formal_evidence_level='E0_UNAVAILABLE', formal_holdout='not_read', paper_claim_ready=False,
        provenance=dict(protocol=dict(path=str(protocol), sha256=protocol_sha256),
            result=dict(path=str(result_path), sha256=sha_file(result_path, budget)),
            run_started=dict(path=str(started_path), sha256=sha_file(started_path, budget)),
            decision=result['decision'], source_commit=plan['source_commit'], source_fingerprints=plan['source_fingerprints'],
            source_protocol=plan['source_protocol'], source_assets=plan['source_assets'], cells=bindings),
        consumer_limits=dict(max_rss_bytes=MAX_RSS, max_wall_seconds=MAX_SECONDS, max_jsonl_line_bytes=MAX_LINE),
        consumer_peak_rss_bytes=budget.peak_rss)
    budget.unchanged()
    require(not list(root.rglob('failure.json')), 'Run acquired failure before completion')
    for method in METHODS:
        equal({path.name for path in (root / method).iterdir()}, {'seed' + str(seed) for seed in SEEDS}, 'Seed inventory changed during read')
        for seed in SEEDS:
            equal({path.name for path in (root / method / ('seed' + str(seed))).iterdir()}, expected_names | {'result.json'}, 'Cell inventory changed during read')
    require(set(path.name for path in root.iterdir()) <= allowed_root, 'Run-root inventory changed during read')
    budget.check()
    return plan, dev, train, report


def statistics_report(plan, dev):
    panels, tails, changes = {}, {}, {}
    for panel, episodes, cases in (('full48', FULL_POINTS, plan['dev_case_ids']), ('probe12', PROBE_POINTS, plan['probe_case_ids'])):
        table = {}
        for method in METHODS:
            scopes = {}
            for scope in ('overall', *('seed' + str(seed) for seed in SEEDS)):
                selected = SEEDS if scope == 'overall' else [int(scope[4:])]
                points = {episode: aggregate([dev[(method, seed)][(episode, case_id)] for seed in selected for case_id in cases]) for episode in episodes}
                if scope == 'overall':
                    for episode in episodes:
                        seed_rewards = [aggregate([dev[(method, seed)][(episode, case_id)] for case_id in cases])['reward'] for seed in SEEDS]
                        points[episode]['seed_reward_sample_sd'] = statistics.stdev(seed_rewards)
                scopes[scope] = points
            table[method] = scopes
            if panel == 'full48':
                tails[method] = {scope: tail_check(points, plan['tail_stability']) for scope, points in scopes.items()}
                changes[method] = {scope: dict(final504_minus_initial0_reward=points[504]['reward'] - points[0]['reward'],
                    initial0=points[0], final504=points[504],
                    reward_improved=points[504]['reward'] > points[0]['reward']) for scope, points in scopes.items()}
        panels[panel] = table
    return dict(panels=panels, tail_stability=dict(criteria=plan['tail_stability'], methods=tails,
        all_methods_and_seeds_passed=all(value['passed'] for scopes in tails.values() for value in scopes.values())),
        final_minus_initial=changes,
        interpretation='Descriptive development results. Tail stability is not learning success, mathematical convergence, independent generalization or a publication guarantee. Negative results are retained.',
        uncertainty='Mean +/- one sample SD across the three seed-level panel reward means (n=3). This is seed variation, not an independent-time confidence interval.',
        probe_scope=plan['probe_scope'], training_scope='Training composition descriptive only; raw scheduled episode rewards and causal trailing24 arithmetic means. No peak selection, interpolation, forced increase or cumulative maximum.')


def write_csv(path, fieldnames, values):
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def trailing(values, window=24):
    queue = deque()
    total = 0.
    for episode in sorted(values):
        queue.append(values[episode]['reward'])
        total += queue[-1]
        if len(queue) > window:
            total -= queue.popleft()
        if len(queue) == window:
            yield episode, total / window


def render(protocol, protocol_sha256, output_root, *, budget=None, allow_synthetic=False):
    budget = budget or Budget(allow_synthetic=allow_synthetic)
    output_root = Path(output_root).absolute()
    require(not output_root.exists(), 'Fresh derived output root required; overwrite forbidden')
    require(output_root.parent.is_dir(), 'Derived parent directory must exist')
    plan, dev, train, report = verify(protocol, protocol_sha256, budget=budget, allow_synthetic=allow_synthetic)
    source_root = Path(plan['run_root']).resolve()
    require(source_root not in output_root.resolve().parents and output_root.resolve() != source_root, 'Derived output must be outside original run')
    require(output_root.resolve() not in source_root.parents, 'Derived output must not contain original run')
    output_root.mkdir(exist_ok=False)
    # All verification precedes any output; source files remain read-only.
    fields = ['method', 'seed', 'episode', 'case_id', 'panels', 'reward', 'request_count', 'completed_requests', 'transferred_bytes', 'steps', 'checkpoint_sha256', 'source_file_sha256']
    extracted = []
    cells = {(cell['method'], cell['seed']): cell for cell in report['provenance']['cells']}
    for method in METHODS:
        for seed in SEEDS:
            for (episode, case_id), value in sorted(dev[(method, seed)].items()):
                panels = (['full48'] if episode in FULL_POINTS else []) + (['probe12'] if case_id in plan['probe_case_ids'] else [])
                cell = cells[(method, seed)]
                extracted.append(dict(method=method, seed=seed, episode=episode, case_id=case_id,
                    panels=';'.join(panels), checkpoint_sha256=cell['files']['ep%03d.pt' % episode]['sha256'],
                    source_file_sha256=cell['files']['dev.jsonl']['sha256'], **value))
    write_csv(output_root / 'dev_rows.csv', fields, extracted)
    training_rows = []
    for method in METHODS:
        for seed in SEEDS:
            means = dict(trailing(train[(method, seed)]))
            for episode, value in sorted(train[(method, seed)].items()):
                training_rows.append(dict(method=method, seed=seed, episode=episode, **value,
                    trailing24_reward=means.get(episode), source_file_sha256=cells[(method, seed)]['files']['training.jsonl']['sha256']))
    write_csv(output_root / 'training_rows.csv', ['method', 'seed', 'episode', 'case_id', 'reward', 'request_count',
        'completed_requests', 'transferred_bytes', 'steps', 'trailing24_reward', 'source_file_sha256'], training_rows)
    aggregate_rows = []
    for panel, methods in report['panels'].items():
        for method, scopes in methods.items():
            for scope, points in scopes.items():
                for episode, value in points.items():
                    aggregate_rows.append(dict(panel=panel, method=method, scope=scope, episode=episode, **value))
    write_csv(output_root / 'panel_summary.csv', ['panel', 'method', 'scope', 'episode', 'reward', 'request_completion_percent',
        'model_kib_per_completed_request', 'request_count', 'completed_requests', 'transferred_bytes', 'row_count', 'seed_reward_sample_sd'], aggregate_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#1676AB', '#499A78', '#B64F86', '#E19427']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .23, 'grid.linewidth': .6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

    def save(fig, name):
        if allow_synthetic:
            fig.text(.5, .5, 'SYNTHETIC — SOFTWARE QA ONLY', ha='center', va='center', rotation=25,
                fontsize=24, color='#9C2633', alpha=.28)
        budget.check()
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output_root / (name + '.' + extension), dpi=240, bbox_inches='tight')
            budget.check()
        plt.close(fig)

    for panel, episodes, title, note in (
        ('full48', FULL_POINTS, 'Fixed full-panel development reward', '48 fixed cases | mean +/- 1 seed SD; n=3 seeds | fixed final: 504'),
        ('probe12', PROBE_POINTS, 'Fixed narrow-probe development reward', '12 fixed cases: families=1, capacity=4096 | mean +/- 1 seed SD; n=3 seeds')):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for method, color in zip(METHODS, colors):
            scopes = report['panels'][panel][method]
            for seed in SEEDS:
                ax.plot(episodes, [scopes['seed' + str(seed)][episode]['reward'] for episode in episodes], color=color, alpha=.22, linewidth=.7)
            means = [scopes['overall'][episode]['reward'] for episode in episodes]
            deviations = [scopes['overall'][episode]['seed_reward_sample_sd'] for episode in episodes]
            ax.fill_between(episodes, [mean - sd for mean, sd in zip(means, deviations)], [mean + sd for mean, sd in zip(means, deviations)], color=color, alpha=.10, linewidth=0)
            ax.plot(episodes, means, color=color, linewidth=1.8, marker='o', markersize=3, label=LABELS[method])
        ax.set(title=('SYNTHETIC | ' if allow_synthetic else '') + title, xlabel='Training episode', ylabel='Mean session reward', xlim=(0, 504))
        ax.set_xticks(FULL_POINTS)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=4, frameon=False, fontsize=9)
        fig.text(.5, -.015, note + '\nSeed SD is not an independent-time confidence interval.', ha='center', fontsize=8, color='#444444')
        save(fig, panel + '_reward')
    for smooth in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, method, color in zip(axes.flat, METHODS, colors):
            for seed in SEEDS:
                values = list(trailing(train[(method, seed)])) if smooth else [(episode, value['reward']) for episode, value in sorted(train[(method, seed)].items())]
                ax.plot([value[0] for value in values], [value[1] for value in values], color=color, alpha={7:1.,17:.66,27:.38}[seed], linewidth=.9, label='seed ' + str(seed))
            ax.set(title=LABELS[method], xlabel='Training episode', ylabel='Session reward', xlim=(1, 504))
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(('SYNTHETIC | ' if allow_synthetic else '') + 'Training composition descriptive only: ' + ('causal trailing24 mean' if smooth else 'raw scheduled episode rewards'), fontsize=12)
        fig.tight_layout(rect=(0, .04, 1, .96))
        fig.text(.5, .015, 'Changing scheduled cases; this is not fixed-panel convergence evidence. No forced monotonic transformation.', ha='center', fontsize=8)
        save(fig, 'training_' + ('trailing24' if smooth else 'raw'))
    report['derived_output_sha256'] = {path.name: sha_file(path, budget) for path in sorted(output_root.iterdir())}
    report['consumer_peak_rss_bytes'] = budget.peak_rss
    with (output_root / 'report.json').open('x') as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    lines = ['# ' + ('SYNTHETIC SOFTWARE QA ONLY | ' if allow_synthetic else '') + 'Fixed 504 development training: complete read-only recovery', '', report['uncertainty'], '', report['interpretation'], '',
        '| Method | Final reward | Reward change (504-0) | All tail scopes pass |', '|---|---:|---:|:---:|']
    for method in METHODS:
        change = report['final_minus_initial'][method]['overall']
        stable = all(value['passed'] for value in report['tail_stability']['methods'][method].values())
        lines.append('| %s | %.6g | %+.6g | %s |' % (LABELS[method], change['final504']['reward'], change['final504_minus_initial0_reward'], stable))
    lines += ['', 'Full per-seed, completion, model-cost, tail and provenance details: report.json; unrounded raw projections: dev_rows.csv and training_rows.csv.', '',
        'The probe12 panel is restricted to one family and capacity4096. It is separate from full48; full-point probe rows are reused. Formal/holdout evidence remains unavailable.']
    with (output_root / 'report.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    with (output_root / 'delivery.json').open('x') as handle:
        budget.unchanged()
        json.dump(dict(schema=SCHEMA, status='SYNTHETIC_SOFTWARE_PREVIEW' if allow_synthetic else 'complete_derived_delivery',
            synthetic_fixture=allow_synthetic, source_protocol_sha256=report['provenance']['protocol']['sha256'],
            files_sha256={path.name: sha_file(path, budget) for path in sorted(output_root.iterdir()) if path.name != 'delivery.json'}),
            handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    budget.check()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('check', 'render'):
        item = sub.add_parser(command)
        item.add_argument('--protocol', type=Path, required=True)
        item.add_argument('--protocol_sha256', help='Optional externally frozen SHA; completed result binding is always checked')
        item.add_argument('--allow_synthetic', action='store_true', help='Software QA only: lower evidence level and watermark every figure; never scientific evidence')
        if command == 'render':
            item.add_argument('--output_root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'check':
        _, _, _, report = verify(args.protocol, args.protocol_sha256, allow_synthetic=args.allow_synthetic)
    else:
        report = render(args.protocol, args.protocol_sha256, args.output_root, allow_synthetic=args.allow_synthetic)
    print(json.dumps(dict(status=report['status'], protocol_sha256=report['provenance']['protocol']['sha256'],
        tail_stability_passed=report['tail_stability']['all_methods_and_seeds_passed'], paper_claim_ready=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
