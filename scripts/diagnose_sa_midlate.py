"""One frozen final64 reference for each of five predeclared SA witnesses.

Requires a pre-existing, hash-bound contract. No training, alternative-policy
rollout, automatic retry or sampling expansion. Uses SHA-bound stage2 per-case
compressed traces for parity; never scans the original multi-GB trace files.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from math import ceil, expm1, log1p
import os
from pathlib import Path
import random
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import joint_stream_same_state_scoring as score
from scripts.diagnose_joint_stream_learning import no_training

WITNESSES = {('sa_ghmappo', 17, 'dev_022'), ('sa_ghmappo', 17, 'dev_046'),
             ('sa_ghmappo', 17, 'dev_017'), ('sa_ghmappo', 7, 'dev_046'),
             ('sa_ghmappo', 27, 'dev_046')}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data, *, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def key(w):
    return (w.get('method', w.get('agent')), int(w['seed']), w['case_id'])


def trace_digest(path):
    h = hashlib.sha256()
    count = 0
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            h.update(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=False, allow_nan=False).encode() + b'\n')
            count += 1
    return dict(canonical_sha256=h.hexdigest(), rows=count)


def prior_metadata(batch, feature, trace):
    """Recompute the frozen public terms using the ORIGINAL first-pass context.

    This is observational arithmetic only; it neither calls feature construction
    again nor supplies alternative logits to the live reference controller.
    """
    from src.envs.core.joint_delivery_planning import fresh_model_span
    from src.envs.core.joint_stream_refinement import state_restore_frames
    from src.envs.core.joint_link_reservation import service_scale
    decision = feature['full_decision']
    s = decision['semantic_state']
    vid = feature['vehicle_id']
    current = s['current_workflow_node']
    candidates = feature['candidates'][0]
    rows = {r['rsu_id']: r for r in s['rsus']}
    rates = s['control_cost_parameters']
    catalog = rates['model_catalog']
    context = feature['controller']
    team_targets = context.get('_team_targets') or {}
    old = (context.get('target_memory') or {}).get(vid)
    raw_mask = [site in decision['control_space']['execution_targets'] or
                (site == 'vehicle' and decision['control_space']['vehicle_execution_available'])
                for site in candidates]
    after_noop = list(raw_mask)
    if any(after_noop[1:]):
        after_noop[0] = False
    old_age = feature['frame'] - old[2] if old else None
    commitment = bool(old and old[0] == current['node_id'] and
                      old[1] in s['reachable_execution_rsu_ids'] and
                      (s['current_node_work_units_progress'] > 0 or
                       s['output_pending'] or old_age < 8))
    reconstructed_mask = [site == old[1] for site in candidates] if commitment else after_noop
    actual = trace['heads'][0]['actual_mask']
    stored = batch['m0'][0].bool().tolist()
    if reconstructed_mask != stored or stored != actual:
        raise ValueError('Execution base/commitment mask reconstruction mismatch')
    denom = max(1., max(4, s['episode_frames_remaining']) / 4)
    model_byte_scale = max(o['resident_size_bytes'] + sum(catalog[x]['resident_size_bytes']
                          for x in o['dependency_ids']) for o in catalog.values())
    queue = {site: sum(work for owner, (target, work) in feature['reservations'].items()
                      if owner != vid and target == site) for site in rows}
    components = []
    for i, site in enumerate(candidates):
        if site not in rows:
            components.append(dict(candidate_index=i, site=site, prior0=float(batch['prior0'][0, i]),
                                   rsu_terms_applicable=False))
            continue
        row = rows[site]
        base = catalog[f"base:{current['required_base_model']}"]
        adapter = catalog[f"adapter:{current['required_adapter']}"]
        alt = dict(s, control_cost_parameters=dict(rates, base_size_bytes=base['transfer_size_bytes'],
                                                  adapter_size_bytes=adapter['transfer_size_bytes']))
        fresh = fresh_model_span(alt, row, current)
        loading = row.get('loading_remaining_frames', {})
        loading_span = sum(loading.get(x, 0) for x in
                           (f"base:{current['required_base_model']}", f"adapter:{current['required_adapter']}"))
        closure = {f"base:{current['required_base_model']}", f"adapter:{current['required_adapter']}"}
        peer = set()
        for owner, state in feature['peer_current_states'].items():
            if owner != vid and team_targets.get(owner) == site and not state['output_pending']:
                node = state['current_workflow_node']
                peer.update((f"base:{node['required_base_model']}", f"adapter:{node['required_adapter']}"))
        overflow = max(0., sum(catalog[x]['resident_size_bytes'] for x in closure | peer) /
                       max(1, row.get('capacity_bytes') or model_byte_scale) - 1)
        conflict = overflow * sum(ceil(catalog[x]['transfer_size_bytes'] / rates['model_bytes_per_frame'])
                                  for x in peer - closure)
        # Frozen prior is built BEFORE f0 is cast to float32. Recompute that
        # path from semantic state; retain stored-feature inversion only as a
        # quantized cross-check, never as an exact attribution.
        state_exact = state_restore_frames(s, site)
        state_unquantized = np.expm1(float(service_scale(state_exact, 480)) * np.log1p(480))
        state_from_feature = expm1(float(batch['f0'][0, i, 23]) * log1p(480))
        reconstructed = np.float32(-(queue.get(site, 0) + state_unquantized) / denom)
        reconstructed = np.float32(reconstructed - (fresh + loading_span + conflict) / denom)
        actual_prior = float(batch['prior0'][0, i])
        if abs(float(reconstructed) - actual_prior) > 2e-6:
            raise ValueError('Public prior decomposition mismatch')
        components.append(dict(candidate_index=i, site=site, rsu_terms_applicable=True,
            queue_service_frames=queue.get(site, 0), state_restore_frames_exact=state_exact,
            state_restore_frames_unquantized_prior_term=float(state_unquantized),
            state_restore_frames_from_float32_feature=state_from_feature,
            cold_fresh_model_frames=fresh, loading_remaining_frames=loading_span,
            peer_family_capacity_conflict_frames=conflict, normalization_denominator=denom,
            reconstructed_prior0=float(reconstructed), prior0=actual_prior,
            ready_base=current['required_base_model'] in row.get('ready_base_model_ids', []),
            ready_adapter=current['required_adapter'] in row.get('cached_adapter_ids', []),
            loading_adapter=current['required_adapter'] in row.get('loading_adapter_ids', []),
            observed_cache_occupied_bytes=row.get('occupied_bytes'), capacity_bytes=row.get('capacity_bytes')))
    old_index = candidates.index(old[1]) if old and old[1] in candidates else None
    return dict(execution_context='first_pass_original_feature_and_reservations',
                raw_physical_mask=raw_mask, after_noop_mask=after_noop,
                feature_mask=stored, actual_execution_mask=actual,
                old_target_memory=deepcopy(old), old_target_memory_age_frames=old_age,
                commitment_applied=commitment,
                commitment_inputs=dict(same_node=bool(old and old[0] == current['node_id']),
                    old_target_reachable=bool(old and old[1] in s['reachable_execution_rsu_ids']),
                    progress=s['current_node_work_units_progress'], output_pending=bool(s['output_pending'])),
                old_target_actual_valid=bool(old_index is not None and actual[old_index]),
                valid_rsu_candidates=[site for i, site in enumerate(candidates) if actual[i] and site in rows],
                public_prior_components=components,
                boundary='Remaining-cost estimate, not a sunk-payment penalty or causal alternate outcome.')


class FixedFrameCapture(score.CaptureTap):
    """Stage3-only collector: fixed frames, with first-pass execution provenance."""
    def __init__(self, agent, witness, frames):
        super().__init__(agent, witness, max_extra_frames=0)
        self.frames = frozenset(frames)
        self._first_features = {}
        self._frame_vehicles = []

    def __enter__(self):
        agent = self.agent
        original_act = agent.act_controls
        original_features = agent._coordinated_features
        original_evaluate = agent._evaluate

        def features(this, decision, frame, reservations):
            answer = original_features(decision, frame, reservations)
            if self._tracing:
                _, candidates, vid, node, work = answer
                self._feature = dict(frame=int(frame), vehicle_id=vid, node_id=node, work=float(work),
                    candidates=deepcopy(candidates), full_decision=deepcopy(decision),
                    decision_sha256=score.json_digest(decision), reservations=deepcopy(reservations),
                    controller=score._controller_context(this),
                    peer_current_states={owner: {name: deepcopy(d['semantic_state'][name]) for name in
                        ('current_workflow_node', 'output_pending')} for owner, d in
                        getattr(this, '_delivery_decisions', {}).items()})
            return answer

        def evaluate(this, data, **kwargs):
            if not self._tracing:
                return original_evaluate(data, **kwargs)
            if self._feature is None or self._feature['frame'] != self._frame:
                raise RuntimeError('Missing fixed-frame feature capture')
            feature = deepcopy(self._feature)
            frozen = score._clone_batch(data)
            result, trace = score.trace_evaluate(this, data, kwargs, call=original_evaluate)
            phase = 'first_execution_pass' if kwargs.get('execution_only', False) else 'second_cache_event_pass'
            vid = feature['vehicle_id']
            if phase == 'first_execution_pass':
                self._first_features[vid] = deepcopy(feature)
            execution_feature = deepcopy(self._first_features[vid])
            actions = result[0].detach().cpu().clone()
            capture = dict(witness=self.witness, frame=self._frame, call_index=len(self._calls),
                phase=phase, feature=feature, execution_feature=execution_feature, batch=frozen,
                reference_actions=actions, reference_trace=trace,
                diagnostic_metadata=prior_metadata(frozen, execution_feature, trace))
            capture['diagnostic_metadata']['base_masks'] = {
                f'm{j}': frozen[f'm{j}'][0].bool().tolist() for j in range(3)}
            capture['diagnostic_metadata']['actual_masks'] = {
                score.HEAD_NAMES[j]: head['actual_mask'] for j, head in enumerate(trace['heads'])}
            if phase == 'second_cache_event_pass':
                capture['forward_rollout'] = {name: result[i].detach().cpu().clone() for name, i in
                    [('actions', 0), ('head_logs', 1), ('entropy', 2), ('values', 3), ('prefix_values', 5),
                     ('active_heads', 6), ('counterfactual_baselines', 7), ('selected_q', 8)]}
                masks = trace['heads']
                capture['diagnostic_metadata']['cache_valid_candidates'] = [
                    feature['candidates'][1][i] for i in masks[1]['valid_candidate_ids']]
                event_candidates = [feature['candidates'][2][i] for i in masks[2]['valid_candidate_ids']]
                capture['diagnostic_metadata']['priority_valid_candidates'] = event_candidates
                capture['diagnostic_metadata']['distinct_prepare_targets'] = list(dict.fromkeys(x[0] for x in event_candidates))
            capture['state_identity'] = dict(batch_sha256=score.tensor_digest(frozen),
                candidates_sha256=score.json_digest(feature['candidates']),
                context_sha256=score.json_digest(feature['controller']),
                decision_sha256=feature['decision_sha256'],
                reference_prefix_sha256=score.tensor_digest({'actions': actions}),
                execution_feature_sha256=score.json_digest(execution_feature))
            self._calls.append(capture)
            return result

        def act(this, decisions, frame, *, deterministic=False):
            if not deterministic:
                raise ValueError('Only final64 greedy reference is authorized')
            if self._tracing:
                raise RuntimeError('Nested reference act_controls')
            self._frame = int(frame)
            self._calls = []
            self._feature = None
            self._first_features = {}
            self._tracing = self._frame in self.frames
            try:
                result = original_act(decisions, frame, deterministic=True)
            finally:
                self._tracing = False
            if self._frame in self.frames:
                counts = Counter((c['feature']['vehicle_id'], c['phase']) for c in self._calls)
                expected = {(vid, phase): 1 for vid in decisions for phase in
                            ('first_execution_pass', 'second_cache_event_pass')}
                if dict(counts) != expected:
                    raise ValueError('Fixed frame omitted/duplicated vehicle or pass')
                self.selected_frames.append(self._frame)
                self.snapshots.extend(self._calls)
            self._calls = []
            return result

        try:
            for name, fn in [('_coordinated_features', features), ('_evaluate', evaluate), ('act_controls', act)]:
                cm = score.patched_method(agent, name, fn)
                cm.__enter__()
                self._contexts.append(cm)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        torch.save(dict(schema='same_state_reference_capture_v1', witness=self.witness,
                        selected_frames=self.selected_frames,
                        selection='stage3 hash-bound fixed midlate frames; all reference vehicles and bothpasses',
                        snapshots=self.snapshots), path)


def verify_contract(contract_path, output_root):
    c = json.loads(contract_path.read_text())
    required = ('schema', 'repo_root', 'stage2_root', 'stage2_result_sha256', 'temporal_plan',
                'temporal_plan_sha256', 'source_run', 'source_protocol_sha256', 'source_commit',
                'script_sha256', 'max_reference_sessions', 'max_reference_frames', 'max_frames_per_session',
                'max_saved_calls', 'max_saved_second_pass_rows', 'new_training_sessions', 'new_optimizer_steps')
    if any(name not in c for name in required):
        raise ValueError('Missing stage3 contract field')
    if c['schema'] != 'stage3_sa_midlate_reference_v1':
        raise ValueError('Unexpected stage3 contract schema')
    caps = dict(max_reference_sessions=5, max_reference_frames=2400, max_frames_per_session=480,
                max_saved_calls=2160, max_saved_second_pass_rows=1080,
                new_training_sessions=0, new_optimizer_steps=0)
    if any(c[name] != value for name, value in caps.items()):
        raise ValueError('Unapproved stage3 budget')
    if c.get('output_root') and Path(c['output_root']).resolve() != output_root:
        raise ValueError('Contract output root mismatch')
    root, stage2, source = [Path(c[name]).resolve() for name in ('repo_root', 'stage2_root', 'source_run')]
    if root != ROOT:
        raise ValueError('Runner must be installed under the declared repository scripts directory')
    required_scripts = {str(Path(__file__).resolve().relative_to(root)),
                        'scripts/joint_stream_same_state_scoring.py', 'scripts/diagnose_joint_stream_learning.py'}
    if not required_scripts.issubset(c['script_sha256']):
        raise ValueError('Contract must bind runner and both imported diagnostic helpers')
    for name, expected in c['script_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen diagnostic script changed: ' + name)
    if sha(stage2 / 'result.json') != c['stage2_result_sha256']:
        raise ValueError('Stage2 result changed')
    planpath = Path(c['temporal_plan']).resolve()
    if sha(planpath) != c['temporal_plan_sha256']:
        raise ValueError('Fixed temporal plan changed')
    plan = json.loads(planpath.read_text())
    stage2result = json.loads((stage2 / 'result.json').read_text())
    if stage2result['status'] != 'complete' or stage2result['reference_sessions'] != 15:
        raise ValueError('Stage2 reference package incomplete')
    if plan['source_result_sha256'] != c['stage2_result_sha256']:
        raise ValueError('Temporal plan is not bound to this stage2 result')
    if len(plan['cases']) != 5 or {key(x['witness']) for x in plan['cases']} != WITNESSES:
        raise ValueError('Fixed five SA witnesses changed')
    if sum(len(x['proposed_capture_frames']) for x in plan['cases']) != 127:
        raise ValueError('Fixed127-frame capture plan changed')
    for item in plan['cases']:
        frames = item['proposed_capture_frames']
        if frames != sorted(set(frames)) or len(frames) > 27 or any(type(f) is not int or not 0 <= f < 480 for f in frames):
            raise ValueError('Invalid fixed frame list')
        regenerated = sorted({f for a in item['proposed_anchors'] if a['frame'] is not None
                              for f in (a['frame'] - 1, a['frame'], a['frame'] + 1) if 0 <= f < 480})
        if frames != regenerated:
            raise ValueError('Frame list differs from frozen anchor rule')
    if sha(source / 'protocol/manifest.json') != c['source_protocol_sha256']:
        raise ValueError('Source protocol changed')
    protocol = json.loads((source / 'protocol/manifest.json').read_text())
    if protocol['source_commit'] != c['source_commit'] or protocol['action_contract'] != 'joint_stream_control_v2':
        raise ValueError('Frozen scientific source identity mismatch')
    if len(protocol['source_fingerprints']) != 123:
        raise ValueError('Expected full123 frozen source fingerprints')
    for name, expected in protocol['source_fingerprints'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen scientific source changed: ' + name)
    for name, expected in protocol['assets'].items():
        if sha(source / 'protocol' / name) != expected:
            raise ValueError('Frozen protocol asset changed: ' + name)
    manifest = json.loads((source / 'sa_ghmappo/manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['git_dirty'] or manifest['source_commit'] != c['source_commit'] or manifest['protocol_sha256'] != c['source_protocol_sha256']:
        raise ValueError('Original SA source manifest invalid')
    if sha(source / 'sa_ghmappo/dev.jsonl') != manifest['output_sha256']['dev.jsonl']:
        raise ValueError('Original SA dev rows changed')
    originals = {}
    for line in (source / 'sa_ghmappo/dev.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['episode'] == 64 and key(row) in WITNESSES:
            originals[key(row)] = row
    if set(originals) != WITNESSES:
        raise ValueError('Missing original final64 SA row')
    for checkpoint in manifest['checkpoints']:
        if checkpoint['seed'] in (7, 17, 27) and sha(source / 'sa_ghmappo' / checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Original SA checkpoint changed')
    refs = {}
    for item in plan['cases']:
        ident = key(item['witness'])
        record = next(x for x in stage2result['results'] if key(x['witness']) == ident)
        capture = next(x for x in stage2result['captures'] if key(x['witness']) == ident)
        parent = Path(capture['path']).parent
        if not parent.is_relative_to(stage2):
            raise ValueError('Stage2 reference path outside declared package')
        checks = {}
        for filename in ('events.jsonl.gz', 'receipts.jsonl.gz', 'reference_row.json'):
            value = sha(parent / filename)
            if value != record['output_sha256'][filename] or value != item['source_hashes'][filename]:
                raise ValueError('Bound stage2 witness file changed: ' + filename)
            checks[filename] = value
        row = json.loads((parent / 'reference_row.json').read_text())
        if row != originals[ident] or not record['reference_summary_exact']:
            raise ValueError('Stage2 witness summary does not match original final64')
        digests = {name: trace_digest(parent / (name + '.gz')) for name in ('events.jsonl', 'receipts.jsonl')}
        if digests != record['reference_traces_exact']:
            raise ValueError('Stage2 compressed trace canonical digest changed')
        refs[ident] = dict(summary=row, digests=digests, compressed_hashes=checks)
    return c, protocol, plan, refs, dict(checked_at=utc(), source_fingerprints_verified=123,
        contract_sha256=sha(contract_path), stage2_result_sha256=c['stage2_result_sha256'],
        temporal_plan_sha256=c['temporal_plan_sha256'], original_sa_manifest_sha256=sha(source / 'sa_ghmappo/manifest.json'),
        source_protocol_sha256=c['source_protocol_sha256'],
        stage2_reference_checks={str(k): v['compressed_hashes'] for k, v in refs.items()},
        raw_trace_policy='Only five SHA-bound stage2 gzip pairs read; no original multi-GB trace scan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    out = args.output_root.resolve()
    if not contract_path.is_file():
        raise ValueError('Pre-existing frozen contract required')
    out.mkdir(parents=True, exist_ok=True)
    # Contract-global claim also prevents replaying this contract into another directory.
    write_json(contract_path.with_name(contract_path.name + '.started.json'),
               dict(started_at=utc(), output_root=str(out), pid=os.getpid(), policy='exclusive; no retry'))
    write_json(out / 'run_started.json', dict(started_at=utc(), contract_path=str(contract_path),
               contract_sha256=sha(contract_path), pid=os.getpid(), training_sessions=0, optimizer_steps=0))
    current = None
    used = {'frames': 0}
    completed = []
    captures = []
    env = None
    try:
        contract, protocol, plan, references, verification = verify_contract(contract_path, out)
        write_json(out / 'verification.json', verification)
        write_json(out / 'command.json', dict(argv=sys.argv, cwd=os.getcwd(), executable=sys.executable))
        from scripts.train_joint_stream_control import make_env, stream_episode
        torch.set_num_threads(1)
        source = Path(contract['source_run'])
        data = torch.load(source / 'protocol/protocol.pt', map_location='cpu', weights_only=False)
        cases = {c['case_id']: c for c in data['cases']['dev']}
        for item in plan['cases']:
            case = cases[item['witness']['case_id']]
            if len(data['sessions'][case['session']]['frames']) > 480:
                raise ValueError('Source session exceeds480-frame cap')
        loaded = {}
        for index, item in enumerate(plan['cases']):
            witness = item['witness']
            method, seed, cid = key(witness)
            current = dict(index=index, method=method, seed=seed, case_id=cid)
            if index >= 5 or used['frames'] + 480 > 2400:
                raise ValueError('Reference session/frame budget exhausted')
            group = out / f'{index:02d}_{method}_seed{seed}_{cid}'
            group.mkdir(exist_ok=False)
            if seed not in loaded:
                loaded[seed] = score.load_exact_models(ROOT, source, method, seed)
            models, provenance = loaded[seed]
            weights = {name: score.module_digest(m) for name, m in models.items()}
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            identity = dict(agent=method, rule=f'{method}_seed{seed}', seed=seed, episode=64, case_id=cid)
            session = {'frames': 0}
            with no_training(models):
                env = make_env(cases[cid], data, protocol)
                original_step = env.step_controls

                def bounded_step(this, controls):
                    if session['frames'] >= 480 or used['frames'] >= 2400:
                        raise ValueError('Physical frame budget exceeded before step')
                    session['frames'] += 1
                    used['frames'] += 1
                    return original_step(controls)

                with score.patched_method(env, 'step_controls', bounded_step):
                    with gzip.open(group / 'events.jsonl.gz', 'xt', encoding='utf8') as ef, gzip.open(group / 'receipts.jsonl.gz', 'xt', encoding='utf8') as rf:
                        with FixedFrameCapture(models['64'], witness, item['proposed_capture_frames']) as tap:
                            row, rolls = stream_episode(env, models['64'], deterministic=True,
                                event_file=ef, receipt_file=rf, identity=identity)
                if row['steps'] != session['frames']:
                    raise ValueError('Physical frame counter/summary mismatch')
                actual = dict(identity, **row)
                expected = references[key(witness)]
                if actual != expected['summary']:
                    write_json(group / 'row_parity_failure.json', dict(actual=actual, expected=expected['summary']))
                    raise ValueError('Full reference summary parity failure')
                trace_checks = {name: trace_digest(group / (name + '.gz')) for name in ('events.jsonl', 'receipts.jsonl')}
                if trace_checks != expected['digests']:
                    raise ValueError('Full reference event/receipt parity failure')
                if tap.selected_frames != item['proposed_capture_frames']:
                    raise ValueError('Did not capture exactly the frozen frame list')
                calls = len(tap.snapshots)
                second = sum(s['phase'] == 'second_cache_event_pass' for s in tap.snapshots)
                if calls > 432 or second > 216 or sum(x['snapshot_count'] for x in completed) + calls > 2160 or sum(x['second_pass_rows'] for x in completed) + second > 1080:
                    raise ValueError('Fixed capture budget exceeded')
                tap.bind_episode_labels(rolls, fleet=8)
                tap.save(group / 'snapshots.pt')
                selected_rolls = [r for r in rolls if int(r['frame_index']) in tap.frames]
                selected_decisions = sum(len(r['actions']) for r in selected_rolls)
                if selected_decisions != second:
                    raise ValueError('Selected rollout/capture row count mismatch')
                torch.save(dict(schema='reference64_selected_rollouts_v1', witness=current,
                    rollouts=selected_rolls, initial_request_count=8, episode_decision_rows=row['decision_rows'],
                    selected_decision_rows=selected_decisions, selected_frames=tap.selected_frames,
                    MC_contract='unchanged physical teamtail/10; teamcredit/8; greedy mass is not behavior probability'),
                    group / 'rollouts.pt')
                with gzip.open(group / 'scoring.jsonl.gz', 'xt', encoding='utf8') as f:
                    for snapshot in tap.snapshots:
                        scored = score.score_snapshot(snapshot, models)
                        scored['diagnostic_metadata'] = snapshot['diagnostic_metadata']
                        f.write(json.dumps(scored, ensure_ascii=False, allow_nan=False) + '\n')
            if weights != {name: score.module_digest(m) for name, m in models.items()}:
                raise ValueError('Reference/scoring changed parameters')
            write_json(group / 'reference_row.json', actual)
            entry = dict(status='pass', completed_at=utc(), witness=current,
                reference_frames=row['steps'], selected_frames=tap.selected_frames,
                snapshot_count=calls, second_pass_rows=second, selected_decision_rows=selected_decisions,
                episode_decision_rows=row['decision_rows'], fleet=8,
                reference_summary_exact=True, reference_traces_exact=trace_checks,
                weights_unchanged=True, capture_and_scoring_rng_guards_passed=True, provenance=provenance,
                output_sha256={name: sha(group / name) for name in ('events.jsonl.gz', 'receipts.jsonl.gz',
                    'snapshots.pt', 'rollouts.pt', 'scoring.jsonl.gz', 'reference_row.json')},
                training_sessions=0, update_calls=0, optimizer_steps=0)
            write_json(group / 'result.json', entry)
            completed.append(entry)
            captures.append(dict(path=str(group / 'snapshots.pt'), sha256=entry['output_sha256']['snapshots.pt'], witness=current))
            write_json(out / 'progress.json', dict(status='running', completed_sessions=len(completed),
                       reference_frames=used['frames'], captures=captures), exclusive=False)
            print(json.dumps(dict(phase='witness_complete', witness=current,
                                  reference_frames=used['frames'], captured_calls=calls)), flush=True)
        final = dict(status='complete', completed_at=utc(), contract_sha256=sha(contract_path),
            reference_sessions=len(completed), reference_frames=used['frames'], max_reference_frames=2400,
            captures=captures, results=completed, snapshot_count=sum(x['snapshot_count'] for x in completed),
            second_pass_rows=sum(x['second_pass_rows'] for x in completed),
            training_sessions=0, update_calls=0, optimizer_steps=0,
            semantic_boundary='fixed final64reference only; other checkpoints are same-state conditional scores, not alternative-policy reward or convergence evidence')
        write_json(out / 'result.json', final)
        write_json(out / 'progress.json', dict(status='complete', completed_sessions=5,
                   reference_frames=used['frames']), exclusive=False)
    except BaseException as exc:
        failure = dict(status='failed', failed_at=utc(), witness=current, error=repr(exc),
            traceback=traceback.format_exc(), completed_sessions=len(completed), charged_reference_frames=used['frames'],
            current_env_frame=getattr(env, 'frame_index', None), training_sessions=0, update_calls=0,
            optimizer_steps=0, policy='preserve all files and exclusive lock; no automatic retry or sample extension')
        if not (out / 'failure.json').exists():
            write_json(out / 'failure.json', failure)
        write_json(out / 'progress.json', failure, exclusive=False)
        raise


if __name__ == '__main__':
    main()
