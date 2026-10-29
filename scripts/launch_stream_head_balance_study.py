"""Claim one diagnosed main-only candidate and collect it without reward selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stream_reward_curve import file_hash, save_new

SEEDS = (7, 17, 27, 37, 47)
METHOD = 'head_balanced_main'
REPLAYS = [dict(method=METHOD, seed=s, case_id=c) for s, c in
           [(7, 'train_014'), (27, 'train_031'), (47, 'train_055')]]


def publish(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def validate_campaign(plan, path, expected_sha):
    from scripts.run_stream_head_balance_study import validate_plan
    if file_hash(path) != expected_sha:
        raise ValueError('prospective plan SHA differs')
    caps = dict(maximum_training_frames=230400, maximum_evaluation_frames=115200,
                maximum_replay_frames=1440, maximum_total_physical_frames=347040,
                maximum_wall_seconds=86400, maximum_disk_bytes=3221225472)
    if any(plan.get(k) != v for k, v in caps.items()):
        raise ValueError('candidate envelope differs')
    if (plan.get('workers') != 2 or plan.get('maximum_scientific_launches') != 1
            or plan.get('replay_episodes') != REPLAYS):
        raise ValueError('worker/launch/replay contract differs')
    if not time.time() < plan['global_deadline_unix'] <= time.time()+86400:
        raise ValueError('candidate absolute deadline differs')
    for seed in SEEDS:
        validate_plan(plan, METHOD, seed, plan['outputs'][METHOD][str(seed)])
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('clean source required before scientific launch')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != plan['source_commit']:
        raise ValueError('frozen scientific commit differs')


def guard(plan):
    if time.time() >= plan['global_deadline_unix']:
        raise RuntimeError('candidate deadline reached; no extension')
    size = sum(p.stat().st_size for p in Path(plan['root']).rglob('*') if p.is_file())
    if size+8*1024**2 > plan['maximum_disk_bytes']:
        raise RuntimeError('candidate disk limit reached')


def terminate_all(active):
    for process in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in active.values():
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def run(path, expected_sha):
    plan = json.loads(path.read_text())
    validate_campaign(plan, path, expected_sha)
    root = Path(plan['root'])
    root.mkdir(parents=True, exist_ok=False)
    save_new(path.parent/'head_balance_execution_claim.json', dict(
        pid=os.getpid(), claimed_at=datetime.now(timezone.utc).isoformat(),
        campaign_sha256=expected_sha, command=sys.argv))
    state = dict(status='running', phase='training_and_fixed_evaluation', pid=os.getpid(),
                 campaign_sha256=expected_sha, source_commit=plan['source_commit'],
                 started_at=datetime.now(timezone.utc).isoformat(), completed_cells=[],
                 active_cells=[], paper_goal_complete=False)
    active, logs, queue = {}, {}, list(SEEDS)

    def start(name, command):
        logs[name] = (root/(name+'.log')).open('x')
        save_new(root/(name+'_command.json'), command)
        active[name] = subprocess.Popen(command, cwd=ROOT, stdout=logs[name],
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))

    def finish(name):
        code = active[name].returncode
        logs.pop(name).close()
        active.pop(name)
        if code != 0:
            raise RuntimeError(f'preserved {name} failure exit{code}; no retry')

    try:
        publish(root/'progress.json', state)
        while queue or active:
            guard(plan)
            while queue and len(active) < plan['workers']:
                seed = queue.pop(0)
                name = f'{METHOD}_seed{seed}'
                start(name, [sys.executable, str(ROOT/'scripts/run_stream_head_balance_study.py'),
                    '--campaign', str(path), '--method', METHOD, '--seed', str(seed),
                    '--output', plan['outputs'][METHOD][str(seed)]])
            for name, process in list(active.items()):
                if process.poll() is not None:
                    finish(name)
                    state['completed_cells'].append(name)
            state['active_cells'] = [dict(name=n, pid=p.pid) for n, p in active.items()]
            publish(root/'progress.json', state)
            time.sleep(5)
        state.update(phase='auditing_replaying_and_comparing', active_cells=[])
        publish(root/'progress.json', state)
        start('collection', [sys.executable, str(ROOT/'scripts/collect_stream_head_balance_study.py'),
            '--campaign', str(path), '--expected_sha', expected_sha])
        while active['collection'].poll() is None:
            guard(plan)
            time.sleep(5)
        finish('collection')
        completion = root/'collection/complete.json'
        if not completion.is_file():
            raise ValueError('collector omitted complete receipt')
        state.update(status='complete', phase='fixed_candidate_evidence_collected',
                     collection_complete=str(completion), collection_sha256=file_hash(completion),
                     completed_at=datetime.now(timezone.utc).isoformat())
        publish(root/'progress.json', state)
        save_new(root/'complete.json', state)
    except BaseException as error:
        terminate_all(active)
        state.update(status='failed_preserved', error=repr(error), traceback=traceback.format_exc(),
                     failed_at=datetime.now(timezone.utc).isoformat())
        publish(root/'progress.json', state)
        save_new(root/'failure.json', state)
        raise
    finally:
        for log in logs.values():
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--expected_sha', required=True)
    args = parser.parse_args()
    run(args.campaign, args.expected_sha)
