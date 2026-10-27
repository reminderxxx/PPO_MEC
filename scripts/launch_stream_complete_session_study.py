"""Freeze one complete-session campaign and detach three unique seed shards."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash,select_train_cases
from scripts.run_stream_complete_session_study import CONTRACT,VARIANTS,load_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent_runs',type=Path,nargs=3,required=True)
    p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--preflight_run',type=Path,required=True)
    p.add_argument('--preflight_audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh campaign root; scientific seeds live in runs/seedN')
    p.add_argument('--continuation',type=Path,required=True)
    a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('clean committed source required before one campaign launch')
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    software=json.loads(a.preflight_audit.read_text())
    assert software['preflight'] and software['verification']['cells']==2
    assert software['verification']['training_episodes']==2 and software['verification']['all_late_releases_observed']
    preflight=json.loads((a.preflight_run/'protocol.json').read_text())
    assert preflight['contract']==CONTRACT and preflight['preflight']
    for field,name in [('main_source_sha256','src/agents/sa_ghmappo_agent.py'),
        ('runner_source_sha256','scripts/run_stream_complete_session_study.py')]:assert file_hash(ROOT/name)==preflight[field]
    assert any(Path(r['path']).resolve()==a.preflight_run.resolve() and
        r['protocol_sha256']==file_hash(a.preflight_run/'protocol.json') for r in software['source_runs'])
    parents,source,manifest,data=load_inputs(a.parent_runs,a.diagnosis)
    cases=select_train_cases(data)
    state=json.loads(a.continuation.read_text())
    assert not state.get('active_scientific_jobs')
    assert state['next_task']['max_new_training_frames']==69120
    assert state['next_task']['status']=='implement_validate_and_commit_before_launch'
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'runs').mkdir()
    now=datetime.now(timezone.utc);plan_path=a.output/'campaign.json'
    seed_outputs={str(s):str((a.output/'runs'/f'seed{s}').resolve()) for s in [7,17,27]}
    plan=dict(contract=CONTRACT,created_at=now.isoformat(),source_commit=git,command=sys.argv,
        seeds=[7,17,27],variants=list(VARIANTS),training_case_ids=[c['case_id'] for c in cases],
        maximum_scientific_launches=1,maximum_training_episodes=144,maximum_training_frames=69120,
        maximum_evaluation_episodes=180,maximum_evaluation_frames=86400,seed_outputs=seed_outputs,
        parent_protocol=str(source),parent_protocol_sha256=file_hash(source),parents=parents,
        diagnosis_sha256=file_hash(a.diagnosis),preflight_run=str(a.preflight_run.resolve()),
        preflight_protocol_sha256=file_hash(a.preflight_run/'protocol.json'),
        preflight_audit=str(a.preflight_audit.resolve()),preflight_audit_sha256=file_hash(a.preflight_audit),
        source_fingerprints={n:file_hash(ROOT/n) for n in [
            'scripts/run_stream_complete_session_study.py','scripts/audit_stream_complete_session_study.py',
            'scripts/replay_stream_complete_session_study.py','scripts/launch_stream_complete_session_study.py',
            'src/agents/sa_ghmappo_agent.py','src/agents/ppo_agent.py',
            'src/envs/core/multi_request_stream_env.py','src/envs/core/joint_stream_rules.py']},
        replay=dict(variant='intent',case_ids=['train_000','train_038','train_067'],
            maximum_episodes=9,maximum_physical_frames=4320),
        stopping='Fixed complete physical sessions and final checkpoints; preserve all outcomes, no extension',
        budget_contract='Equal external sessions and maximum horizons, actual frame and gradient counts may differ',
        formal_holdout='not read',paper_claim_ready=False)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    children=[]
    try:
        for seed in plan['seeds']:
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_stream_complete_session_study.py'),
                '--parent_runs',*[str(r.resolve()) for r in a.parent_runs],
                '--diagnosis',str(a.diagnosis.resolve()),'--seed',str(seed),
                '--campaign',str(plan_path.resolve()),'--output',seed_outputs[str(seed)]]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=a.output/f'seed{seed}.log'
            with log.open('xb') as stream:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(dict(seed=seed,pid=child.pid,command=cmd,output=seed_outputs[str(seed)],log=str(log.resolve())))
            (a.output/'launcher.json').write_text(json.dumps(dict(launched_at=now.isoformat(),children=children),indent=2)+'\n')
    except Exception:
        import traceback
        (a.output/'launch_failure.json').write_text(json.dumps(dict(failed_at=datetime.now(timezone.utc).isoformat(),
            traceback=traceback.format_exc(),children=children,automatic_retry=False),indent=2)+'\n')
        raise
    state['completed_intent_children']=state.get('children',[])
    state.update(stage='complete_session_confirmation_running',updated_at=now.isoformat(),
        scientific_git=git,complete_session_campaign=str(a.output.resolve()),
        complete_session_plan_sha256=file_hash(plan_path),children=children,active_scientific_jobs=children,
        next_check_not_before=(now+timedelta(minutes=20)).isoformat(),last_checked_at=now.isoformat())
    state['next_task']['status']='running'
    a.continuation.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(campaign=str(a.output.resolve()),source_commit=git,
        children=[dict(seed=c['seed'],pid=c['pid']) for c in children],next_check_not_before=state['next_check_not_before'])))


if __name__=='__main__':main()
