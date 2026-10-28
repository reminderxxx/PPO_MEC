"""One claimed campaign with bounded workers and automatic audit/figure collection."""
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
from scripts.run_stream_reward_curve import METHODS, SEEDS, file_hash, save_new, validate_plan

REPLAY_EPISODES = [dict(method=m,seed=s,case_id=c) for m,s,c in [
    ('main',7,'train_000'),('main',17,'train_038'),('main',27,'train_067'),
    ('mappo',7,'train_000'),('ppo',17,'train_038'),('dqn',27,'train_067'),
    ('mappo',37,'train_067'),('ppo',47,'train_000'),('dqn',37,'train_038')]]


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def check_campaign(plan, campaign_path, expected_sha):
    if file_hash(campaign_path) != expected_sha: raise ValueError('frozen campaign SHA differs')
    if (plan['maximum_total_physical_frames'] != 1393920 or plan['maximum_artifact_bytes'] != 10737418240
            or plan['maximum_rule_frames'] != 7200 or plan['maximum_replay_frames'] != 4320
            or plan['maximum_scientific_launches'] != 1 or plan['parallel_workers'] != 3
            or plan['replay_episodes'] != REPLAY_EPISODES
            or plan['probe_case_ids'] != ['train_000','train_038','train_067']):
        raise ValueError('campaign envelope or prospective replay pool differs')
    if not time.time() < plan['global_deadline_unix'] <= time.time()+172800:
        raise ValueError('invalid prospective global deadline')
    for method in METHODS:
        for seed in SEEDS: validate_plan(plan, method, seed, plan['outputs'][method][str(seed)])
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('clean frozen tracked source required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() != plan['source_commit']:
        raise ValueError('campaign scientific commit differs')


def run(campaign_path, expected_sha):
    plan = json.loads(campaign_path.read_text()); check_campaign(plan,campaign_path,expected_sha)
    root = Path(plan['root']); root.mkdir(parents=True,exist_ok=False)
    save_new(campaign_path.parent/'campaign_execution_claim.json',dict(pid=os.getpid(),
        claimed_at=datetime.now(timezone.utc).isoformat(),campaign_sha256=expected_sha,command=sys.argv))
    state = dict(status='running',phase='training_and_fixed_evaluation',pid=os.getpid(),
        campaign_sha256=expected_sha,source_commit=plan['source_commit'],root=str(root),completed_cells=[],
        active_cells=[],started_at=datetime.now(timezone.utc).isoformat(),next_check_not_before=None,
        training_frames=0,evaluation_frames=0,paper_goal_complete=False)
    progress = root/'progress.json'; active = {}; logs = {}; queue = [(m,s) for s in SEEDS for m in METHODS]
    def guard():
        if time.time() >= plan['global_deadline_unix']: raise RuntimeError('48hour campaign deadline reached')
        size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if size+8*1024**2 > plan['maximum_artifact_bytes']: raise RuntimeError('campaign artifact cap reached')
    def start(name, command):
        log=(root/(name+'.log')).open('x');logs[name]=log
        save_new(root/(name+'_command.json'),command)
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        active[name]=process
    def stop():
        for process in active.values():
            if process.poll() is None:
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        for process in active.values():
            if process.poll() is None:
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
    try:
        publish(progress,state)
        while queue or active:
            guard()
            while queue and len(active) < plan['parallel_workers']:
                method,seed=queue.pop(0);name=f'{method}_seed{seed}'
                start(name,[sys.executable,str(ROOT/'scripts/run_stream_reward_curve.py'),'--campaign',str(campaign_path),
                    '--method',method,'--seed',str(seed),'--output',plan['outputs'][method][str(seed)]])
            for name, process in list(active.items()):
                code=process.poll()
                if code is None: continue
                logs.pop(name).close();active.pop(name)
                if code != 0: raise RuntimeError(f'preserved cell failure: {name}, exit{code}; no retry')
                state['completed_cells'].append(name)
            state['active_cells']=[dict(name=n,pid=p.pid) for n,p in active.items()]
            totals={'training_frames':0,'evaluation_frames':0}
            for method in METHODS:
                for seed in SEEDS:
                    home=Path(plan['outputs'][method][str(seed)])
                    path=home/'complete.json' if (home/'complete.json').exists() else home/'manifest.json'
                    if path.exists():
                        try:
                            record=json.loads(path.read_text())
                            for k in totals: totals[k]+=record[k]
                        except json.JSONDecodeError: pass  # Live progress is not a terminal scientific receipt.
            state.update(totals); publish(progress,state)
            time.sleep(2)
        state['phase']='auditing_replaying_and_exporting';publish(progress,state)
        name='collection';start(name,[sys.executable,str(ROOT/'scripts/collect_stream_reward_curve.py'),
            '--campaign',str(campaign_path),'--expected_sha',expected_sha])
        while active[name].poll() is None: guard();time.sleep(2)
        code=active[name].returncode;logs.pop(name).close();active.pop(name)
        if code != 0: raise RuntimeError('preserved collection failure; no rerun or reward selection')
        collection=json.loads((root/'collection/complete.json').read_text())
        state.update(status='complete',phase='reward_curve_delivered',collection=collection,
            completed_at=datetime.now(timezone.utc).isoformat())
        publish(progress,state);save_new(root/'complete.json',state)
    except BaseException as error:
        stop();state.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc(),
            failed_at=datetime.now(timezone.utc).isoformat())
        publish(progress,state);save_new(root/'failure.json',state);raise
    finally:
        for log in logs.values(): log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--expected_sha',required=True);args=parser.parse_args();run(args.campaign,args.expected_sha)
