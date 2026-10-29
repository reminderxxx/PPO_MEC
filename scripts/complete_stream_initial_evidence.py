"""Complete only the missing frozen initial-policy evaluations (never train).

The existing 15 initial probes, 60 fixed finals and 12 deterministic rule records
are references to one immutable audited package. Only 45 missing initial cases
and three declared initial replays interact with the physical simulator.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stream_reward_curve import (CellBudget, GzipRecords, GuardedEnv,
    file_hash, json_text, loaded_policy, tensor_hash, utc_now)
from scripts.train_joint_stream_control import make_env
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent

CONTRACT = 'stream_initial_evidence_v1'
PARENT_CAMPAIGN_SHA = 'abd965ec2617cae544ed38f9cf607f9c9c87383e324dda8211783f46824357e1'
PARENT_AUDIT_SHA = 'd795a848637b7828214cd41582387075e4ad50a7269682a4d756674165f1a948'
PARENT_SOURCE = 'e00d5faf9c3f73a689174237245ed6f85394b2f2'
SEEDS = [7, 17, 27, 37, 47]
CASES = ['train_000', 'train_014', 'train_019', 'train_007', 'train_024',
         'train_038', 'train_043', 'train_031', 'train_048', 'train_062',
         'train_067', 'train_055']
PROBES = ['train_000', 'train_038', 'train_067']
REPLAYS = [dict(seed=7, case_id='train_014'), dict(seed=27, case_id='train_031'),
           dict(seed=47, case_id='train_055')]
EVENT_KEYS = ('frame_index', 'vehicle_id', 'request_id', 'workflow_id', 'status',
    'control', 'control_planning', 'cache_request', 'state_transfer_link_frames',
    'execution_rsu_id', 'service_target_changed', 'association_handoff')
METRICS = ('reward', 'completed_requests', 'state_transfer_bytes',
    'state_transfer_link_frames', 'model_transfer_link_frames',
    'input_transfer_bytes', 'input_transfer_link_frames', 'output_transfer_bytes',
    'output_transfer_link_frames', 'control_intent_transmitted_bytes',
    'shared_state_transmitted_bytes', 'handoff_failed_count',
    'association_handoff_count', 'service_target_change_count')
FRAME_SUM_METRICS = ('state_transfer_bytes', 'state_transfer_link_frames',
    'model_transfer_link_frames', 'input_transfer_bytes', 'input_transfer_link_frames',
    'output_transfer_bytes', 'output_transfer_link_frames',
    'control_intent_transmitted_bytes', 'shared_state_transmitted_bytes')
REQUIRED_SOURCES = {'scripts/complete_stream_initial_evidence.py',
    'scripts/audit_stream_initial_evidence.py', 'scripts/run_stream_reward_curve.py',
    'scripts/train_joint_stream_control.py', 'src/agents/sa_ghmappo_agent.py',
    'src/agents/ppo_agent.py', 'src/envs/core/multi_request_stream_env.py'}
COMPATIBLE_SOURCE_PATHS = ['src/agents/ppo_agent.py', 'src/agents/sa_ghmappo_agent.py']
COMPATIBILITY_ASSERTIONS = {'legacy_default_forward_bitwise_equal',
    'legacy_default_update_bitwise_equal', 'legacy_initialization_tensor_equal'}


def canonical_hash(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def validate_plan(plan, *, root=ROOT, verify_files=True):
    if (plan.get('contract') != CONTRACT or plan.get('seeds') != SEEDS
            or plan.get('training_case_ids') != CASES or plan.get('probe_case_ids') != PROBES
            or plan.get('initial_replay_episodes') != REPLAYS):
        raise ValueError('fixed initial evidence matrix/replays required')
    if any(plan.get(k, 0) != 0 for k in ('training_frames', 'fit_calls', 'downloads', 'hidden_episodes')):
        raise ValueError('zero training/fitting/download/hidden budget required')
    b = plan.get('budgets', {})
    for key, maximum in [('maximum_total_physical_frames', 23040),
                         ('maximum_wall_seconds', 10800),
                         ('maximum_output_bytes', 536870912)]:
        if type(b.get(key)) is not int or not 0 < b[key] <= maximum:
            raise ValueError('invalid hard cap: ' + key)
    if b.get('maximum_frames_per_episode') != 480:
        raise ValueError('480-frame complete episode cap required')
    output = Path(plan['output']).resolve()
    if Path(plan['audit_output']).resolve() != output / 'audit':
        raise ValueError('consumer output must remain inside total output budget')
    if output == ROOT or ROOT in output.parents and output == ROOT / 'scripts':
        raise ValueError('invalid scientific output path')
    for k, expected in [('parent_campaign', PARENT_CAMPAIGN_SHA), ('parent_audit', PARENT_AUDIT_SHA)]:
        if plan[k].get('sha256') != expected:
            raise ValueError('only the frozen parent package may be referenced')
        if verify_files and file_hash(plan[k]['path']) != expected:
            raise ValueError('parent bytes changed: ' + k)
    old_root = Path(plan['parent_audit']['path']).resolve().parent.parent
    if output == old_root or old_root in output.parents:
        raise ValueError('old scientific package must remain immutable')
    if verify_files:
        fingerprints = plan.get('source_fingerprints', {})
        if not REQUIRED_SOURCES <= set(fingerprints):
            raise ValueError('incomplete execution source inventory')
        for name, sha in fingerprints.items():
            path = (Path(root) / name).resolve()
            if Path(root).resolve() not in path.parents or file_hash(path) != sha:
                raise ValueError('execution source hash mismatch: ' + name)
    return plan


def row_reference(audit_path, audit_sha, pointer, row):
    return dict(source_path=str(Path(audit_path).resolve()), source_sha256=audit_sha,
                pointer=pointer, row_sha256=canonical_hash(row), record=row)


def source_compatibility(plan, parent, *, root=ROOT):
    """Permit only two approved opt-in source additions after recorded bitwise QA.

    A new execution commit is always disclosed. This does not claim that new
    source bytes are identical to the historical campaign or validate a new
    algorithm against scientific evaluation results.
    """
    changed = {}
    for name, old in parent['source_fingerprints'].items():
        path = (Path(root) / name).resolve()
        if Path(root).resolve() not in path.parents:
            raise ValueError('original source path escapes execution root')
        current = file_hash(path)
        if current != old:
            if name not in COMPATIBLE_SOURCE_PATHS:
                raise ValueError('original scientific source changed: ' + name)
            if plan['source_fingerprints'].get(name) != current:
                raise ValueError('new compatible source not pinned by execution plan')
            changed[name] = dict(old_sha256=old, new_sha256=current)
    approval = plan.get('source_compatibility')
    if not changed and approval is None:
        return None
    if not isinstance(approval, dict) or approval.get('approved_changed_paths') != COMPATIBLE_SOURCE_PATHS:
        raise ValueError('exact two-path source compatibility approval required')
    ref = approval.get('qualification', {})
    if not ref.get('path') or file_hash(ref['path']) != ref.get('sha256'):
        raise ValueError('source compatibility qualification bytes differ')
    proof = json.loads(Path(ref['path']).read_text())
    validation = proof.get('validation', {})
    assertions = proof.get('assertions', {})
    if (proof.get('contract') != 'stream_initial_legacy_source_compatibility_v1'
            or proof.get('status') != 'passed' or proof.get('parent_source_commit') != PARENT_SOURCE
            or proof.get('changes') != changed or not changed
            or set(assertions) != COMPATIBILITY_ASSERTIONS or any(value is not True for value in assertions.values())
            or not validation.get('command') or type(validation.get('passed')) is not int
            or validation['passed'] <= 0 or validation.get('failures') != 0):
        raise ValueError('source compatibility qualification incomplete or mismatched')
    return dict(qualification=deepcopy(ref), contract=proof['contract'], status=proof['status'],
                approved_changed_paths=approval['approved_changed_paths'], changes=changed,
                assertions=assertions, validation=validation,
                boundary='Synthetic default-path compatibility only; new source bytes and commit remain distinct.')


def load_bindings(plan, *, verify_files=True):
    """Authenticate immutable input references; do not re-run any parent audit."""
    validate_plan(plan, verify_files=verify_files)
    parent = json.loads(Path(plan['parent_campaign']['path']).read_text())
    audited = json.loads(Path(plan['parent_audit']['path']).read_text())
    if (parent.get('source_commit') != PARENT_SOURCE or parent.get('seeds') != SEEDS
            or parent.get('training_case_ids') != CASES or parent.get('probe_case_ids') != PROBES
            or audited.get('campaign_sha256') != PARENT_CAMPAIGN_SHA
            or audited.get('source_commit') != PARENT_SOURCE
            or audited.get('status') != 'curve_receipts_and_checkpoints_audited'
            or audited.get('final_selection') != 'episode96_fixed_no_peak'
            or audited.get('training_case_ids') != CASES or audited.get('probe_case_ids') != PROBES):
        raise ValueError('parent protocol, pool or fixed-final identity differs')
    pool = [(c['method'], c['seed']) for c in audited['cells']]
    if len(pool) != 20 or set(pool) != {(m, s) for m in ['main', 'mappo', 'ppo', 'dqn'] for s in SEEDS}:
        raise ValueError('parent audited method/seed matrix incomplete')
    protocol = parent['protocol']
    compatible = None
    if verify_files:
        for path_key, sha_key in [('path', 'sha256'), ('protocol_pt', 'protocol_pt_sha256')]:
            if file_hash(protocol[path_key]) != protocol[sha_key]:
                raise ValueError('parent input protocol bytes differ')
        compatible = source_compatibility(plan, parent)
    reused, finals, checkpoints, rules = [], [], [], []
    for seed in SEEDS:
        i, cell = next((i, c) for i, c in enumerate(audited['cells'])
                       if c['method'] == 'main' and c['seed'] == seed)
        initial = [r for r in cell['checkpoints'] if r['episode'] == 0]
        if len(initial) != 1:
            raise ValueError('one original ep000 checkpoint required')
        cp = initial[0]
        expected_path = Path(parent['outputs']['main'][str(seed)]) / 'checkpoints/ep000.pt'
        if (Path(cp['path']).resolve() != expected_path.resolve() or cp['method'] != 'main'
                or cp['seed'] != seed or cp['policy_class'] != IntentAwareStreamSAGHMAPPOAgent.__name__
                or cp['native_method'] != 'sa_ghmappo'
                or cp['checkpoint_contract'] != IntentAwareStreamSAGHMAPPOAgent.checkpoint_contract
                or cp['cumulative_training_frames'] != 0 or cp['cumulative_updates'] != 0
                or cp['policy_initialization'] != 'random_from_scratch_no_policy_parent'):
            raise ValueError('initial checkpoint provenance differs')
        if verify_files and file_hash(cp['path']) != cp['sha256']:
            raise ValueError('original ep000 checkpoint bytes changed')
        checkpoints.append(deepcopy(cp))
        fixed_final = [r for r in cell['checkpoints'] if r['episode'] == 96]
        if len(fixed_final) != 1:
            raise ValueError('one original fixed ep096 checkpoint required')
        for episode, expected_cases, target in [(0, PROBES, reused), (96, CASES, finals)]:
            rows = [(j, r) for j, r in enumerate(cell['evaluation']) if r['episode'] == episode]
            if len(rows) != len(expected_cases) or {r['case_id'] for _, r in rows} != set(expected_cases):
                raise ValueError('parent initial/final case matrix differs')
            for cid in expected_cases:
                j, row = next((j, r) for j, r in rows if r['case_id'] == cid)
                if (row['method'] != 'main' or row['seed'] != seed or not row['physical_terminal']
                        or row['checkpoint_episode'] != episode
                        or row['checkpoint_sha256'] != (cp if episode == 0 else fixed_final[0])['sha256']
                        or row['is_curve_probe'] != (cid in PROBES)
                        or row['is_fixed_final'] != (episode == 96)):
                    raise ValueError('parent evaluation checkpoint identity differs')
                target.append(row_reference(plan['parent_audit']['path'], PARENT_AUDIT_SHA,
                                             f'/cells/{i}/evaluation/{j}', deepcopy(row)))
    if len(audited['rules']) != 12 or {r['case_id'] for r in audited['rules']} != set(CASES):
        raise ValueError('one immutable deterministic rule record per case required')
    for cid in CASES:
        j, row = next((j, r) for j, r in enumerate(audited['rules']) if r['case_id'] == cid)
        rules.append(row_reference(plan['parent_audit']['path'], PARENT_AUDIT_SHA,
                                   f'/rules/{j}', deepcopy(row)))
    references = dict(parent_campaign=deepcopy(plan['parent_campaign']), parent_audit=deepcopy(plan['parent_audit']),
        protocol=deepcopy(protocol), initial_checkpoints=checkpoints, reused_initial=reused,
        final_reference=finals, gain_reference=rules, network_requests=0,
        parent_audit_reruns=0, reused_evaluations=87, new_independent_samples_claimed=False,
        historical_source=dict(commit=parent['source_commit'], fingerprints=parent['source_fingerprints']),
        execution_source=dict(commit=plan['source_commit'], fingerprints=plan['source_fingerprints']),
        source_compatibility=compatible)
    return parent, references


def selected_cases(data):
    rows = data['cases']['train']
    cases = {r['case_id']: r for r in rows if r['case_id'] in CASES}
    if len(cases) != 12 or sum(r['case_id'] in CASES for r in rows) != 12:
        raise ValueError('twelve unique original consumed-train cases required')
    for case in cases.values():
        session = data['sessions'][case['session']]
        if session['split'] != 'train' or not 0 < len(session['frames']) <= 480:
            raise ValueError('consumed-train complete physical session required')
    return cases


def budget_for(output, plan):
    b = plan['budgets']
    return CellBudget(output, dict(maximum_frames_per_episode=480,
        maximum_training_frames_per_cell=0,
        maximum_evaluation_frames_per_cell=b['maximum_total_physical_frames'],
        maximum_cell_wall_seconds=b['maximum_wall_seconds'],
        maximum_cell_disk_bytes=b['maximum_output_bytes']))


def save_bounded(path, value, budget):
    raw = json_text(value).encode()
    budget.check(len(raw))
    with Path(path).open('xb') as stream:
        stream.write(raw)


def append_bounded(stream, value, budget):
    raw = json_text(value)
    budget.check(len(raw.encode()))
    stream.write(raw)
    stream.flush()


def schedule_identity(case):
    return sorted([dict(request_id=a.request_id, vehicle_id=a.vehicle_id, frame=a.frame,
                        total_nodes=len(a.workflow.nodes)) for a in case['arrivals']],
                  key=lambda r: (r['frame'], r['vehicle_id'], r['request_id']))


def episode(env, agent, budget, identity, writer):
    """Record literal native events/commands, including idle frames and late waves."""
    guarded = GuardedEnv(env, budget, 'evaluation')
    agent.reset_controller()
    before_tensor = tensor_hash(agent)
    frames = hashlib.sha256(); controls = hashlib.sha256(); literal = hashlib.sha256()
    total = 0.; decisions = 0
    while not env._control_episode_done():
        f = env.frame_index
        commands, roll = agent.act_controls(env.decision_states(), f, deterministic=True)
        commands_copy = {vehicle: asdict(control) if is_dataclass(control) else deepcopy(control)
                         for vehicle, control in commands.items()}
        _, reward, done, info = guarded.step_controls(commands)
        parts = info['frame_cost_receipt']['reward_components']
        if not math.isfinite(reward) or abs(reward - sum(parts.values())) > 1e-8:
            raise ValueError('native reward components differ')
        native_events = deepcopy(info['events'])
        events = deepcopy(native_events)
        for event in events:
            event['control_planning'] = deepcopy(agent.last_diagnostics['by_vehicle'][event['vehicle_id']])
            controls.update(json_text({k: event[k] for k in EVENT_KEYS if k in event}).encode())
        body = dict(frame_index=f, reward=reward, reward_components=parts,
            receipt=deepcopy(info['frame_cost_receipt']), commands=commands_copy,
            native_events=native_events, events=events, frame_after_step=env.frame_index,
            terminal_after_step=bool(env._control_episode_done()),
            done_returned=bool(done), released_request_ids=sorted(env._released))
        frames.update(json_text({k: body[k] for k in ('frame_index', 'reward', 'reward_components')}).encode())
        literal.update(json_text(body).encode())
        writer.write_record(dict(identity, record_kind='native_frame', **body))
        total += reward
        if roll is not None:
            decisions += len(roll['actions'])
    if (tensor_hash(agent) != before_tensor or int(agent.completed_updates) != 0
            or agent.optimizer.state or getattr(agent, 'surrogate_optimizer', None).state):
        raise ValueError('evaluation mutated policy or optimization state')
    if env._released != {a.request_id for a in env.arrivals}:
        raise ValueError('late scheduled arrivals not completely released')
    summary = env.summary()
    if summary['steps'] != guarded.receipt_count:
        raise ValueError('physical step receipts differ')
    return dict(identity, **summary, reward=total, decision_rows=decisions,
        physical_terminal=True, bootstrap_team_tail=0.,
        frame_digest=frames.hexdigest(), control_digest=controls.hexdigest(),
        literal_digest=literal.hexdigest(), scheduled_requests=schedule_identity({'arrivals': env.arrivals}),
        last_scheduled_release_frame=max(a.frame for a in env.arrivals),
        released_request_ids=sorted(env._released), horizon_frames=len(env.frames))


def run(plan_path, expected_sha, *, verify=True, audit_after=True):
    plan_path = Path(plan_path)
    if file_hash(plan_path) != expected_sha:
        raise ValueError('prospective plan SHA mismatch')
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, verify_files=verify)
    if verify:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        if dirty or commit != plan['source_commit']:
            raise ValueError('clean frozen execution commit required')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    budget = budget_for(output, plan)
    writer = None
    started = time.monotonic()
    try:
        save_bounded(output / 'launch_claim.json', dict(contract=CONTRACT, pid=os.getpid(),
            started_at=utc_now(), started_unix=time.time(), plan_path=str(plan_path.resolve()), plan_sha256=expected_sha,
            source_commit=plan['source_commit'], command=sys.argv), budget)
        parent, references = load_bindings(plan, verify_files=verify)
        save_bounded(output / 'protocol.json', dict(plan=plan, plan_sha256=expected_sha), budget)
        save_bounded(output / 'source_references.json', references, budget)
        source = json.loads(Path(parent['protocol']['path']).read_text())
        data = torch.load(parent['protocol']['protocol_pt'], map_location='cpu', weights_only=False)
        cases = selected_cases(data)
        cp_by = {cp['seed']: cp for cp in references['initial_checkpoints']}
        writer = GzipRecords(output / 'frames.jsonl.gz', budget)
        records = []
        with (output / 'episodes.jsonl').open('x') as rows:
            for phase, items in [('initial', [dict(seed=s, case_id=c) for s in SEEDS for c in CASES if c not in PROBES]),
                                 ('replay', REPLAYS)]:
                for item in items:
                    cp = cp_by[item['seed']]
                    agent = loaded_policy(Path(cp['path']), cp, IntentAwareStreamSAGHMAPPOAgent)
                    identity = dict(method='main', seed=item['seed'], case_id=item['case_id'], phase=phase,
                        checkpoint_episode=0, checkpoint_sha256=cp['sha256'])
                    row = episode(make_env(cases[item['case_id']], data, source), agent, budget, identity, writer)
                    records.append(row)
                    append_bounded(rows, row, budget)
        writer.close(); writer = None
        if len(records) != 48 or budget.frames['train'] != 0:
            raise ValueError('new physical episode matrix differs')
        inventory = {name: file_hash(output / name) for name in
                     ['launch_claim.json', 'protocol.json', 'source_references.json', 'episodes.jsonl', 'frames.jsonl.gz']}
        complete = dict(status='initial_acquisition_complete', contract=CONTRACT, plan_sha256=expected_sha,
            source_commit=plan['source_commit'], initial_episodes=45, replay_episodes=3,
            actual_physical_frames=budget.frames['evaluation'], training_frames=0, fit_calls=0,
            elapsed_seconds=time.monotonic() - started, output_sha256=inventory,
            paper_goal_complete=False, independent_generalization_claimed=False)
        save_bounded(output / 'complete.json', complete, budget)
        if audit_after:
            from scripts.audit_stream_initial_evidence import consume
            return consume(plan_path, expected_sha, verify=verify)
        return complete
    except BaseException as exc:
        if writer is not None:
            try: writer.close()
            except BaseException: pass
        failure = dict(status='failed_preserved_no_retry', error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(), plan_sha256=expected_sha,
            actual_physical_frames=budget.frames['evaluation'], training_frames=0, fit_calls=0,
            elapsed_seconds=time.monotonic()-started, failed_at=utc_now(), paper_goal_complete=False)
        with (output / 'failure.json').open('x') as stream:
            stream.write(json_text(failure))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected_sha', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.plan, args.expected_sha)))


if __name__ == '__main__':
    main()
