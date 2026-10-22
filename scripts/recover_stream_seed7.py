"""Recover only saved seed7 observer parity from a failed diagnostic.

``contract`` prints a separate, byte-bound recovery contract. ``check`` is a
read-only stdlib preflight. ``run`` validates existing files into a fresh derived
directory; it never imports torch, constructs a model/environment, or trains.
The failed original run and the incomplete three-seed matrix remain failed and
incomplete. This entry does not relax the completed-matrix summary contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

SCHEMA = 'stream_seed7_partial_recovery_v1'
SOURCE_COMMIT = '8c836f9edbaedb42e99dd3026ea6935796533ba0'
STATUS = 'RECOVERED_SEED7_ONLY_OBSERVER_PARITY'
HEADS = ('execution', 'cache', 'prepare_priority')
KINDS = ('behavior', 'credit', 'minibatch', 'update')
LIMITS = dict(max_rss_bytes=512 * 1024 ** 2, max_wall_seconds=1800,
              max_cpu_seconds=1500, max_line_bytes=16 * 1024 ** 2,
              max_audit_bytes=6 * 1024 ** 3, max_audit_rows=100000)
HERE = Path(__file__).resolve()
VALIDATOR = HERE.with_name('sa_stage5c_credit_audit.py')
EPISODE_FIELDS = ('episode', 'case_id', 'summary_sha256', 'rolls_sha256',
                  'receipts_sha256', 'rng_after_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    out = {}
    for key, value in items:
        require(key not in out, 'Duplicate JSON key')
        out[key] = value
    return out


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= 32 * 1024 ** 2, 'Receipt unexpectedly large')
    return parse(path.read_bytes())


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


class Resources:
    def __init__(self, *, wall=time.monotonic, cpu=time.process_time, rss=None):
        self.wall = wall; self.cpu = cpu; self.rss = rss or self.current_rss
        self.started = wall(); self.cpu_started = cpu(); self.peak = 0

    @staticmethod
    def current_rss():
        used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return int(used)
        if sys.platform.startswith('linux'):
            return int(used) * 1024
        raise RuntimeError('Unsupported RSS units')

    def check(self):
        self.peak = max(self.peak, self.rss())
        require(self.peak <= LIMITS['max_rss_bytes'], 'Recovery exceeded 512 MiB RSS')
        require(self.wall() - self.started <= LIMITS['max_wall_seconds'], 'Recovery wall limit exceeded')
        require(self.cpu() - self.cpu_started <= LIMITS['max_cpu_seconds'], 'Recovery CPU limit exceeded')


def sha_file(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
            if guard:
                guard.check()
    return digest.hexdigest()


def binding(path, guard=None):
    path = Path(path).resolve()
    require(path.is_file(), 'Missing input: ' + str(path))
    before = path.stat()
    digest = sha_file(path, guard)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed while hashing')
    return dict(path=str(path), size_bytes=after.st_size, sha256=digest)


def required_inputs(root, original_contract):
    paths = {'original_contract': Path(original_contract).resolve(),
             'failure': root / 'failure.json', 'progress': root / 'progress.json',
             'run_started': root / 'run_started.json',
             'off_result': root / 'seed7/off/result.json',
             'audit': root / 'seed7/on/audit/events.jsonl'}
    for side in ('off', 'on'):
        for name in ('checkpoint.pt', 'final_evidence.pt'):
            paths[side + '_' + name.replace('.', '_')] = root / 'seed7' / side / name
        for episode in range(1, 5):
            paths['%s_episode%d' % (side, episode)] = root / 'seed7' / side / ('episode%d.json' % episode)
    return paths


def fresh_output(root, output):
    require(not output.exists(), 'Recovery output must be fresh; no retry/overwrite')
    require(root != output and root not in output.parents, 'No output inside failed original root')
    require(output not in root.parents, 'Output cannot contain the original root')
    require(output.parent.is_dir(), 'Recovery output parent must exist')


def make_contract(qualification, original_contract, output, guard=None):
    """Read/hash only; caller explicitly saves the printed JSON outside old root."""
    guard = guard or Resources(); guard.check()
    root = Path(qualification).resolve(); output = Path(output).resolve()
    fresh_output(root, output)
    paths = required_inputs(root, original_contract)
    original = read_json(paths['original_contract'])
    paths['protocol_manifest'] = Path(original['protocol']['path']).resolve()
    contract = dict(schema=SCHEMA, status='frozen', seed=7, recovered_seeds=[7],
        not_run_seeds=[17, 27], qualification_root=str(root), output_root=str(output),
        original_source_commit=SOURCE_COMMIT, limits=LIMITS.copy(),
        counts=dict(training_sessions=0, training_updates=0, models=0, environments=0),
        operation_scope='existing bytes/JSON/schema joins only; no torch deserialization or scientific execution',
        input_bindings={name: binding(path, guard) for name, path in paths.items()},
        recovery_sources={name: binding(path, guard) for name, path in
                          (('entry', HERE), ('validator', VALIDATOR))})
    verify_receipts(contract)
    return contract


def _path(contract, name):
    return Path(contract['input_bindings'][name]['path'])


def verify_receipts(contract):
    root = Path(contract['qualification_root'])
    old = read_json(_path(contract, 'original_contract'))
    require(old['schema'] == 'stream_stability_diagnostic_v1' and old['status'] == 'frozen', 'Original frozen contract required')
    require(old['source_commit'] == SOURCE_COMMIT == contract['original_source_commit'], 'Historical source differs')
    require(Path(old['output_root']).resolve() == root, 'Original output identity differs')
    require(old['seeds'] == [7, 17, 27] and old['sides'] == ['off', 'on'] and
            type(old['training_budget']) is int and old['training_budget'] == 64 and
            type(old['sessions_per_seed']) is int and old['sessions_per_seed'] == 4 and
            type(old['epochs']) is int and old['epochs'] == 4, 'Original64/first4 scope differs')
    require(old['limits']['audit_bytes_per_seed'] == LIMITS['max_audit_bytes'] and
            old['limits']['audit_rows_per_seed'] == LIMITS['max_audit_rows'], 'Original audit caps differ')
    for absent in ('result.json', 'seed7/result.json', 'seed7/on/result.json', 'seed17', 'seed27'):
        require(not (root / absent).exists(), 'Expected absent original result/seed now exists: ' + absent)
    require(not (root / 'seed7/on/audit/failure.json').exists(), 'Producer audit failure receipt exists')
    failure = read_json(_path(contract, 'failure')); progress = read_json(_path(contract, 'progress'))
    started = read_json(_path(contract, 'run_started')); off = read_json(_path(contract, 'off_result'))
    require(failure['status'] == 'failed' and failure['completed_seeds'] == 0 and
            failure['no_retry'] is True and failure['exception_type'] == 'RuntimeError' and
            failure['message'] == 'Frozen RSS limit exceeded; no automatic expansion/retry', 'Unexpected original failure')
    expected_context = dict(seed=7, side='on', episode=4, case_id=off['case_ids'][3], phase='independent_consumer')
    require(failure['context'] == expected_context == progress['context'], 'Failure was not seed7 independent consumer')
    counts = failure['counts']
    require(type(counts['training_sessions']) is int and counts['training_sessions'] == 8 and
            type(counts['training_updates']) is int and counts['training_updates'] == 2 and
            counts == progress['counts'], 'Failed run count scope differs')
    original_sha = contract['input_bindings']['original_contract']['sha256']
    require(started['contract_sha256'] == original_sha and started['source_commit'] == SOURCE_COMMIT and
            started['no_retry'] is True and type(started['pid']) is int and
            type(started['started_at']) is str, 'Original start binding differs')
    protocol_binding = contract['input_bindings']['protocol_manifest']
    require(protocol_binding['sha256'] == old['protocol']['sha256'] and
            Path(old['protocol']['path']).resolve() == Path(protocol_binding['path']), 'Original protocol binding differs')
    protocol = read_json(protocol_binding['path'])
    require(protocol['protocol'] == 'joint_stream_learning_v2' and
            protocol['action_contract'] == 'joint_stream_control_v2' and protocol['smoke'] is False and
            protocol['episodes'] == 64 and protocol['seeds'] == [7, 17, 27], 'Original stream protocol differs')
    cases = protocol['training_schedule']['7'][:4]
    require(len(protocol['training_schedule']['7']) == 64 and len(set(cases)) == 4, 'Original first4 schedule differs')
    require(off['seed'] == 7 and off['side'] == 'off' and off['episode_count'] == 4 and
            off['case_ids'] == cases, 'Off receipt seed/sessions/cases differ')
    off_counts = off['original_call_counts']
    require(off_counts['training_sessions'] == 4 and off_counts['training_updates'] == 1, 'Off count scope differs')
    for key in ('training_sessions', 'training_updates', 'training_frames', 'environments',
                'adam_calls', 'backward_calls', 'categorical_sample_calls', 'evaluate_calls'):
        require(counts[key] == 2 * off_counts[key], 'Original off/on operation count mismatch: ' + key)
    require(counts['models'] == 2, 'Expected exactly two original SA learner instances')
    for name in ('checkpoint_pt', 'final_evidence_pt'):
        a = contract['input_bindings']['off_' + name]; b = contract['input_bindings']['on_' + name]
        require(a['size_bytes'] == b['size_bytes'] and a['sha256'] == b['sha256'], 'Saved off/on payload bytes differ: ' + name)
        field = 'checkpoint_sha256' if name == 'checkpoint_pt' else 'final_evidence_sha256'
        require(off[field] == a['sha256'], 'Off result payload SHA differs')
    for episode, case in enumerate(cases, 1):
        a = read_json(_path(contract, 'off_episode%d' % episode)); b = read_json(_path(contract, 'on_episode%d' % episode))
        require(a['episode'] == episode and a['case_id'] == case, 'Episode/order/case differs')
        require(all(a[field] == b[field] for field in EPISODE_FIELDS), 'Archived episode digest parity differs')
        for field in EPISODE_FIELDS[2:]:
            require(type(a[field]) is str and len(a[field]) == 64 and all(c in '0123456789abcdef' for c in a[field]), 'Invalid episode digest')
    return dict(original=old, failure=failure, progress=progress, off=off, cases=cases)


def check_contract(path, guard=None):
    """No outputs, project imports, deserialization or scientific execution."""
    guard = guard or Resources(); guard.check()
    path = Path(path).resolve(); contract = read_json(path)
    require(contract['schema'] == SCHEMA and contract['status'] == 'frozen' and
            type(contract['seed']) is int and contract['seed'] == 7 and
            contract['recovered_seeds'] == [7] and contract['not_run_seeds'] == [17, 27], 'Seed7 partial recovery contract required')
    require(contract['limits'] == LIMITS and all(type(v) is int for v in contract['limits'].values()), 'Frozen recovery limits differ')
    require(contract['counts'] == dict(training_sessions=0, training_updates=0, models=0, environments=0) and
            all(type(v) is int for v in contract['counts'].values()), 'Recovery cannot add scientific budget')
    root = Path(contract['qualification_root']).resolve(); output = Path(contract['output_root']).resolve()
    fresh_output(root, output)
    require(root != path and root not in path.parents, 'Recovery contract must be outside old root')
    paths = required_inputs(root, _path(contract, 'original_contract'))
    old = read_json(paths['original_contract']); paths['protocol_manifest'] = Path(old['protocol']['path']).resolve()
    require(set(paths) == set(contract['input_bindings']), 'Missing/extra recovery input binding')
    for name, expected_path in paths.items():
        supplied = contract['input_bindings'][name]
        require(Path(supplied['path']) == expected_path.resolve(), 'Bound input path differs: ' + name)
        require(binding(expected_path, guard) == supplied, 'Bound input size/SHA differs: ' + name)
    require(set(contract['recovery_sources']) == {'entry', 'validator'}, 'Recovery source bindings missing')
    for name, source in (('entry', HERE), ('validator', VALIDATOR)):
        require(binding(source, guard) == contract['recovery_sources'][name], 'Recovery implementation changed: ' + name)
    verify_receipts(contract); guard.check()
    return contract


class Stats:
    __slots__ = ('n', 'mean', 'm2', 'lo', 'hi', 'negative', 'zero', 'positive')
    def __init__(self):
        self.n = 0; self.mean = self.m2 = 0.; self.lo = math.inf; self.hi = -math.inf
        self.negative = self.zero = self.positive = 0
    def add(self, value):
        require(type(value) in (int, float) and math.isfinite(value), 'Invalid summary number')
        value = float(value); self.n += 1; delta = value - self.mean
        self.mean += delta / self.n; self.m2 += delta * (value - self.mean)
        self.lo = min(self.lo, value); self.hi = max(self.hi, value)
        self.negative += value < 0; self.zero += value == 0; self.positive += value > 0
    def result(self):
        return dict(n=self.n, mean=self.mean if self.n else None,
            sd=math.sqrt(max(0., self.m2 / (self.n - 1))) if self.n > 1 else None,
            min=self.lo if self.n else None, max=self.hi if self.n else None,
            negative=self.negative, zero=self.zero, positive=self.positive)


def metrics(): return defaultdict(Stats)
def metric_result(values): return {k: v.result() for k, v in sorted(values.items())}


def summarize_audit(contract, guard):
    """One raw-hash/terminal/statistics pass; retain only small UID projections."""
    path = _path(contract, 'audit'); behaviors = {}; credits = {}; credited = set(); sampled = set()
    per_head = {name: {scope: metrics() for scope in ('behavior', 'active', 'forced')} for name in HEADS}
    per_case = defaultdict(lambda: {name: metrics() for name in HEADS})
    minibatch = metrics(); optimizers = defaultdict(metrics); update_stats = metrics()
    counts = Counter(); sample_counts = Counter(); cases = Counter(); updates = set()
    digest = hashlib.sha256(); full = hashlib.sha256(); total = 0; header = terminal = None
    old = read_json(_path(contract, 'original_contract'))
    off = read_json(_path(contract, 'off_result'))
    with path.open('rb') as handle:
        index = 0
        while True:
            raw = handle.readline(LIMITS['max_line_bytes'] + 1)
            if not raw: break
            guard.check()
            require(len(raw) <= LIMITS['max_line_bytes'] and raw.endswith(b'\n'), 'Oversize/truncated audit row')
            require(terminal is None, 'Data after terminal')
            record = parse(raw); p = record['payload']; kind = record['event_type']
            require(record['seq'] == index and type(record['seq']) is int, 'Audit sequence differs')
            if index == 0:
                require(kind == 'header', 'Missing audit header'); header = p
                require(p['evidence_kind'] == 'unverified_actual_chain' and p['actual_training_verified'] is False,
                        'Synthetic or self-certified audit origin')
                require(p['source_commit'] == SOURCE_COMMIT and p['max_bytes'] == LIMITS['max_audit_bytes'] and
                        p['max_rows'] == LIMITS['max_audit_rows'], 'Header source/limits differ')
                provenance = p['provenance']
                require(provenance['contract_sha256'] == contract['input_bindings']['original_contract']['sha256'] and
                        provenance['protocol_sha256'] == old['protocol']['sha256'] and
                        Path(provenance['protocol_path']).resolve() == Path(old['protocol']['path']).resolve() and
                        provenance['seed'] == 7 and provenance['side'] == 'on' and
                        provenance['source'] == 'fresh_original_stochastic_stream_sessions', 'Header provenance differs')
                require(record['run_uid'] == p['run_uid'], 'Header run UID differs')
            elif kind == 'terminal':
                terminal = p
                require(p['status'] == 'complete_unverified' and p['actual_training_verified'] is False,
                        'Complete unverified actual-chain terminal required')
                require(p['rows_before'] == index and p['bytes_before'] == total and
                        p['data_sha256'] == digest.hexdigest() and p['event_counts'] == dict(counts),
                        'Raw terminal counts/hash differ')
            else:
                require(record['run_uid'] == header['run_uid'] and p['seed'] == 7 and kind in KINDS, 'Crossed run/seed/event')
                counts[kind] += 1
                if kind == 'behavior':
                    uid = p['behavior_uid']; require(uid not in behaviors, 'Duplicate behavior UID')
                    case = p['metadata']['session']['case_id']; masks = p['masks']
                    active = tuple(sum(mask) > 1 for mask in masks); B = tuple(p['metadata']['stored_baseline'])
                    require(case in off['case_ids'] and len(B) == 3, 'Behavior case/baseline shape differs')
                    behaviors[uid] = (case, active, B); cases[case] += 1
                    for j, name in enumerate(HEADS):
                        m = per_head[name]['behavior']
                        for key, value in dict(candidate_count=len(p['candidates'][j]), legal_count=sum(masks[j]),
                                active=int(active[j]), old_log_probability=p['old_log_probs'][j],
                                old_probability=math.exp(p['old_log_probs'][j]), stored_baseline=B[j]).items(): m[key].add(value)
                elif kind == 'credit':
                    bid = p['behavior_uid']; cid = p['credit_uid']
                    require(bid in behaviors and bid not in credited and cid not in credits, 'Credit UID join differs')
                    credited.add(bid); credits[cid] = bid; case, active, B = behaviors[bid]
                    require(tuple(p['B']) == B and tuple(p['active_heads']) == active and p['fleet'] == 8, 'Stored baseline/mask/fleet differs')
                    require(abs(p['T_team'] - p['T'] / 8) <= 2e-6 * max(1., abs(p['T_team'])) and
                            p['global_value_target'] == p['T_team'], 'MC/team units differ')
                    for j, name in enumerate(HEADS):
                        error = p['B'][j] - p['T_team']
                        require(abs(p['raw_adv'][j] - (p['T_team'] - p['B'][j])) <=
                                2e-5 * max(1., abs(p['raw_adv'][j])), 'Raw advantage arithmetic differs')
                        values = dict(team_target=p['T_team'], baseline=p['B'][j], baseline_error=error,
                            baseline_squared_error=error * error, raw_advantage=p['raw_adv'][j], normalized_advantage=p['normalized_adv'][j])
                        for key, value in values.items():
                            per_head[name]['active' if active[j] else 'forced'][key].add(value)
                            if active[j]: per_case[case][name][key].add(value)
                elif kind == 'minibatch':
                    sample_counts['minibatches'] += 1; sample_counts['early_stop_minibatches'] += bool(p['kl_early_stop'])
                    minibatch['kl'].add(p['kl']); meta = p['metadata']; scores = meta['scores']
                    require(len(p['behavior_uids']) == len(p['credit_uids']), 'Minibatch join length differs')
                    for bid, cid in zip(p['behavior_uids'], p['credit_uids']):
                        require(bid in behaviors and credits.get(cid) == bid, 'Minibatch foreign key differs'); sampled.add(bid)
                    n = len(p['behavior_uids']); sample_counts['sample_occurrences'] += n
                    require(len(scores['ratio']) == len(scores['active_heads']) == n, 'Minibatch ratio shape differs')
                    for row, flags in zip(scores['ratio'], scores['active_heads']):
                        require(len(row) == len(flags) == 3, 'Minibatch head shape differs')
                        for j, enabled in enumerate(flags):
                            if enabled:
                                minibatch['active_ratio_' + HEADS[j]].add(row[j])
                                minibatch['active_clipped_' + HEADS[j]].add(int(row[j] < .8 or row[j] > 1.2))
                    for call in meta['optimizer_calls']:
                        label = call['optimizer']; require(label in ('main', 'surrogate'), 'Unknown optimizer')
                        sample_counts[label + '_adam_calls'] += 1
                        for group, values in call['postclip_gradients'].items():
                            for key, value in values.items(): optimizers[label][group + '.postclip.' + key].add(value)
                        for group, values in call['parameter_delta'].items():
                            for key, value in values.items(): optimizers[label][group + '.actual_delta.' + key].add(value)
                    losses = meta['losses']
                    require(losses is None and not p['step_observed'] or type(losses) is dict, 'Invalid minibatch loss observation')
                    for key, value in (losses or {}).items(): minibatch['loss.' + key].add(value)
                else:
                    require(p['update_uid'] not in updates, 'Duplicate update UID'); updates.add(p['update_uid'])
                    for key, value in p['metadata']['losses'].items():
                        if type(value) in (int, float): update_stats[key].add(value)
            if kind != 'terminal': digest.update(raw)
            full.update(raw); total += len(raw); index += 1
            require(total <= LIMITS['max_audit_bytes'] and index <= LIMITS['max_audit_rows'], 'Audit exceeds frozen cap')
    require(header is not None and terminal is not None, 'Missing header/terminal')
    require(full.hexdigest() == contract['input_bindings']['audit']['sha256'] and
            total == contract['input_bindings']['audit']['size_bytes'], 'Audit differs from frozen input')
    require(len(updates) == 1 and set(behaviors) == credited and set(cases) == set(off['case_ids']), 'Partial update/case coverage')
    require(counts['behavior'] == off['coverage']['decision_rows'], 'Off decision coverage differs')
    for j, name in enumerate(HEADS):
        require(per_head[name]['behavior']['active'].positive == off['coverage']['active_heads'][j] and
                per_head[name]['behavior']['active'].zero == off['coverage']['forced_heads'][j], 'Off active/forced coverage differs')
    require(sample_counts['main_adam_calls'] == off['coverage']['main_adam_calls'] and
            sample_counts['surrogate_adam_calls'] == off['coverage']['surrogate_adam_calls'], 'Off optimizer coverage differs')
    return dict(seed=7, event_counts=dict(counts), rows=index, bytes=total, file_sha256=full.hexdigest(),
        terminal=terminal, behavior_rows_by_case=dict(cases),
        counts=dict(behavior_rows=len(behaviors), credit_rows=len(credited), updates=len(updates),
            sampled_unique_behavior_rows=len(sampled), never_sampled_behavior_rows=len(behaviors) - len(sampled), **sample_counts),
        head={name: {scope: metric_result(m) for scope, m in groups.items()} for name, groups in per_head.items()},
        active_credit_by_case={case: {name: metric_result(m) for name, m in heads.items()} for case, heads in sorted(per_case.items())},
        minibatch=metric_result(minibatch), optimizer={name: metric_result(m) for name, m in optimizers.items()}, update=metric_result(update_stats))


def load_validator():
    spec = importlib.util.spec_from_file_location('seed7_recovery_schema_validator', VALIDATOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@contextmanager
def bounded_validator(module, guard):
    original = module._Joins.event
    def checked(this, *args, **kwargs):
        guard.check(); result = original(this, *args, **kwargs); guard.check(); return result
    module._Joins.event = checked
    try: yield
    finally: module._Joins.event = original


def analysis_text(result):
    summary = result['seed7_summary']; c = summary['counts']
    lines = ['# seed7 部分观察证据回收', '',
        '原诊断仍为 FAILED，三 seed 矩阵未完成；seed17、seed27 为 NOT_RUN。本次新增训练会话与更新均为0。', '',
        '独立读取已保存文件：off/on 的 checkpoint.pt 和 final_evidence.pt 全字节 SHA 相同，4会话的原始精确摘要/RNG回执一致。未反序列化任何 Torch 文件。完整 schema join 与原始 terminal/hash 已核验，但 complete_unverified 不证明随机训练来源，actual_training_verified 保持 false。', '',
        'seed7 首4会话、首1次 learner 更新：行为 %d 行，信用 %d 行，minibatch %d 次，主/辅助 Adam %d/%d 次，KL早停 %d 次。' %
        (c['behavior_rows'], c['credit_rows'], c['minibatches'], c['main_adam_calls'], c['surrogate_adam_calls'], c['early_stop_minibatches']), '']
    coverage = result['recorded_off_coverage']
    lines += ['已绑定原 off 回执：idle 帧=%s；更新前后 Adam 状态/完成计数=%s。这些是原回执观察值，本工具不重新加载优化器验证。' %
        (coverage.get('idle_frames', 'UNVERIFIED'), coverage.get('updates', 'UNVERIFIED')), '']
    for head in HEADS:
        m = summary['head'][head]['active'].get('baseline_squared_error', {})
        lines.append('- %s：自由信用%d行；baseline 对本次事实团队 MC 目标的描述性 MSE=%s。' % (head, m.get('n', 0), m.get('mean')))
    lines += ['', '这些统计保留 seed7 和 case 差异；重复 epoch/head 观察不是独立样本。中间逐参数梯度和 Adam 向量的 off/on 字节级 trace 未保存，无法补证。最终文件一致与回执一致的范围不等同于完整中间执行逐位一致。', '',
        '本回收不定位长期退化原因，不预言 seed17 走势，不认定算法修复、收敛或正式论文证据晋级。原来源身份取自冻结合同与运行回执，未经本工具独立重新执行认证。', '']
    return '\n'.join(lines)


def run(contract_path, guard=None):
    guard = guard or Resources(); contract_binding = binding(contract_path, guard)
    contract = check_contract(contract_path, guard)
    require(binding(contract_path, guard) == contract_binding, 'Recovery contract changed during preflight')
    contract_sha = contract_binding['sha256']
    output = Path(contract['output_root']); output.mkdir(exist_ok=False)
    write_json(output / 'contract.json', contract)
    write_json(output / 'run_started.json', dict(schema=SCHEMA, started_at=datetime.now(timezone.utc).isoformat(),
        contract_sha256=contract_sha, counts=dict(training_sessions=0, training_updates=0)))
    try:
        # Size-bounded JSON/hash pass precedes the library's line iterator.
        summary = summarize_audit(contract, guard)
        module = load_validator()
        with bounded_validator(module, guard):
            audit = module.validate(_path(contract, 'audit'), max_file_bytes=LIMITS['max_audit_bytes'], compact_joins=True)
        guard.check()
        require(audit['status'] == 'complete_unverified' and audit['structural_complete'] is True and
                audit['actual_training_verified'] is False and audit['synthetic_fixture'] is False, 'Compact schema validation failed')
        for field in ('rows', 'bytes', 'event_counts'):
            require(audit[field] == summary[field], 'Independent schema/summary accounting differs: ' + field)
        require(audit['observed_terminal'] == summary['terminal'] and
                audit['observed_main_optimizer_steps'] == summary['counts']['main_adam_calls'] and
                audit['observed_surrogate_optimizer_steps'] == summary['counts']['surrogate_adam_calls'], 'Independent optimizer/terminal accounting differs')
        # A fixed final hash pass catches input changes during both consumers.
        for name, supplied in contract['input_bindings'].items():
            require(binding(supplied['path'], guard) == supplied, 'Input changed during recovery: ' + name)
        for name, supplied in contract['recovery_sources'].items():
            require(binding(supplied['path'], guard) == supplied, 'Recovery code changed during execution: ' + name)
        require(binding(contract_path, guard) == contract_binding, 'Recovery contract changed during execution')
        state = verify_receipts(contract); guard.check()
        result = dict(schema=SCHEMA, status=STATUS, recovered_at=datetime.now(timezone.utc).isoformat(),
            original_run_status='failed', original_run_failed=True, full_matrix_incomplete=True,
            recovered_seeds=[7], not_run_seeds=[17, 27], seed_status={'7': STATUS, '17': 'NOT_RUN', '27': 'NOT_RUN'},
            counts=dict(training_sessions=0, training_updates=0), original_run_counts=state['failure']['counts'],
            qualification_root=contract['qualification_root'], recovery_contract_sha256=contract_sha,
            original_source_commit=SOURCE_COMMIT, input_bindings=contract['input_bindings'], recovery_sources=contract['recovery_sources'],
            saved_final_files_byte_equal=True, four_episode_exact_digest_parity=True,
            parity_scope='saved whole-file bytes and archived exact episode digests; intermediate gradient/Adam byte traces unavailable',
            actual_training_verified=False, source_stochastic_origin='UNVERIFIED_by_this_recovery',
            formal_holdout='not_read', evidence_level='bounded seed7 diagnostic only; formal E0_UNAVAILABLE',
            convergence='UNVERIFIED', causal_root_cause='UNIDENTIFIED', algorithm_fix='NOT_CLAIMED',
            compact_schema_validation=audit, seed7_summary=summary, resource_limits=LIMITS,
            recorded_off_coverage=state['off']['coverage'],
            recorded_off_coverage_scope='archived off receipt; active/forced/Adam counts cross-checked to on audit; idle/empty-Adam not re-executed',
            peak_process_rss_bytes=guard.peak)
        write_json(output / 'summary.json', summary)
        with (output / 'analysis.md').open('x') as handle: handle.write(analysis_text(result))
        write_json(output / 'result.json', result)
        return result
    except BaseException as exc:
        write_json(output / 'failure.json', dict(schema=SCHEMA, status='failed', original_run_status='failed',
            recovered_seeds=[], not_run_seeds=[17, 27], counts=dict(training_sessions=0, training_updates=0),
            exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('contract', help='Print byte-bound JSON; does not write files')
    prepare.add_argument('--qualification', type=Path, required=True)
    prepare.add_argument('--original-contract', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    for name in ('check', 'run'):
        sub.add_parser(name).add_argument('--contract', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'contract':
        value = make_contract(args.qualification, args.original_contract, args.output)
    elif args.command == 'check':
        value = check_contract(args.contract)
        value = dict(status='RECOVERY_CHECK_PASSED', output_root=value['output_root'], counts=value['counts'])
    else:
        value = run(args.contract)
        value = {key: value[key] for key in ('status', 'recovered_seeds', 'not_run_seeds', 'counts')}
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
