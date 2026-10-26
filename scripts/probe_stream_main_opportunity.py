"""Train-only, score-blind opportunity probe of the frozen paper stream method.

Existing rules and physical costs remain immutable. This is development evidence,
not a formal benchmark or a counterfactual oracle. Full frame receipts are retained.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_joint_stream_control import make_env
from src.agents.ppo_agent import StreamControlPPO
from src.envs.core.joint_stream_rules import StreamCacheRule


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_train_cases(data):
    """Twelve input-selected conditions, four per train session; no scores read."""
    selected = []
    conditions = [
        ('short', 'queued_batch', 1, 4096),
        ('medium', 'stream', 2, 4096),
        ('long', 'queued_batch', 2, 8192),
        ('short', 'stream', 2, 8192),
    ]
    for session in sorted({c['session'] for c in data['cases']['train']}):
        for size, mode, families, capacity in conditions:
            selected.append(next(c for c in data['cases']['train']
                if (c['session'], c['size'], c['arrival_mode'], c['families'], c['capacity'])
                == (session, size, mode, families, capacity)))
    assert len(selected) == 12 and all(data['sessions'][c['session']]['split'] == 'train' for c in selected)
    return selected


def run_episode(env, agent, trace_path, *, learned):
    if learned:
        agent.reset_controller()
    total = 0.
    components = Counter()
    statuses = Counter()
    counts = Counter()
    start = time.monotonic()
    digest = hashlib.sha256()
    with Path(trace_path).open('x') as trace:
        while not env._control_episode_done():
            decisions = env.decision_states()
            if learned:
                controls, roll = agent.act_controls(decisions, env.frame_index, deterministic=True)
                if roll is not None:
                    counts['decision_rows'] += len(roll['actions'])
                    for j in range(3):
                        counts[f'active_head_{j}'] += int(roll['active_heads'][:, j].sum())
                del roll
            else:
                controls = agent.act(decisions, env.frame_index)
            _, reward, _, info = env.step_controls(controls)
            receipt = info['frame_cost_receipt']
            assert abs(reward - sum(receipt['reward_components'].values())) < 1e-8
            total += reward
            components.update(receipt['reward_components'])
            for e in info['events']:
                statuses[e['status']] += 1
                counts['association_handoff'] += int(e['association_handoff'])
                counts['service_target_changed'] += int(e['service_target_changed'])
                counts['prepare_commands'] += int(e['control']['prepare_target'] is not None)
                counts['cache_commands'] += int(e['control']['cache_target'] is not None)
                counts['model_admitted'] += len(e['cache_request'].get('admitted_object_ids', []))
                counts['model_evicted'] += len(e['cache_request'].get('evicted_object_ids', []))
                counts['cache_command_failed'] += int(e['cache_request']['status'] in (
                    'capacity_blocked', 'blocked_current_model_dependency', 'link_queue'))
            packed = json.dumps(dict(frame_index=env.frame_index - 1, reward=reward,
                receipt=receipt, events=info['events']), sort_keys=True)
            trace.write(packed + '\n')
            digest.update((packed + '\n').encode())
    return dict(reward=total, components=dict(components), statuses=dict(statuses),
        counts=dict(counts), **env.summary(), trace_sha256=digest.hexdigest(),
        wall_seconds=time.monotonic() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--checkpoint_root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(a.protocol.read_text())
    assert manifest['source_commit'].startswith('35975cb3')
    assert file_hash(a.protocol.parent / 'protocol.pt') == manifest['assets']['protocol.pt']
    data = torch.load(a.protocol.parent / 'protocol.pt', weights_only=False)
    cases = select_train_cases(data)
    # Preflight budget depends only on runtime. Scientific phase keeps all 12 cases.
    if a.preflight:
        cases = cases[:1]
    methods = ['reactive', 'naive_prefetch', 'gain_prefetch', 'sa_seed7', 'sa_seed17', 'sa_seed27']
    if a.preflight:
        methods = ['reactive', 'sa_seed7']
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(), phase='train_only_opportunity',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        parent_protocol_sha256=file_hash(a.protocol), parent_protocol=str(a.protocol.resolve()),
        cases=[{k:c[k] for k in ['case_id','session','size','arrival_mode','families','capacity']} for c in cases],
        methods=methods, preflight=a.preflight, predeclared='all cases and all methods retained; no reward-based selection',
        claim='bounded training diagnosis only; compatible existing rule pool, not cross-contract global strongest',
        frozen_baseline_sha256=file_hash(ROOT/'src/envs/core/joint_stream_rules.py'),
        frozen_environment_sha256=file_hash(ROOT/'src/envs/core/multi_request_stream_env.py'),
        command=sys.argv, checkpoints={})
    for method in methods:
        if method.startswith('sa_'):
            seed=int(method.removeprefix('sa_seed'))
            f=a.checkpoint_root / f'sa_ghmappo_seed{seed}_ep64.pt'
            protocol['checkpoints'][method]=dict(path=str(f),sha256=file_hash(f))
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (a.output/'results.jsonl').open('x') as results:
        for method in methods:
            for case in cases:
                agent=(StreamControlPPO.from_checkpoint(protocol['checkpoints'][method]['path'])
                       if method.startswith('sa_') else StreamCacheRule(method))
                path=a.output/f'{method}_{case["case_id"]}.jsonl'
                row=run_episode(make_env(case,data,manifest),agent,path,learned=method.startswith('sa_'))
                results.write(json.dumps(dict(method=method,case_id=case['case_id'],**row))+'\n')
                results.flush()
                print(json.dumps(dict(method=method,case_id=case['case_id'],reward=row['reward'],
                    completed=row['completed_requests'],seconds=round(row['wall_seconds'],2))),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',
        results_sha256=file_hash(a.output/'results.jsonl'),rows=len(methods)*len(cases)),indent=2)+'\n')


if __name__ == '__main__':
    main()
