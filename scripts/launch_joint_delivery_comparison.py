"""One clean-source, matched-budget delivery comparison, including audits and support."""
import argparse,json,platform,subprocess,sys,time,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import _hash_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output_root',type=Path,default=ROOT/'artifacts/joint_delivery_control');a=p.parse_args();os.chdir(ROOT);root=a.output_root.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'requires clean source'
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();root.mkdir(parents=True,exist_ok=False)
    py=sys.executable
    def run(command,log):
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with log.open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    run([py,'scripts/prepare_joint_delivery_control.py','--output_root',str(root/'protocol')],root/'prepare_command.log')
    prepared=root/'protocol/manifest.json';m=json.loads(prepared.read_text())
    primary=['sa_ghmappo','mappo','sa_no_prediction','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl'];supports=['sa_no_surrogate','sa_fixed_entropy'];names=primary+supports
    files=list((ROOT/'src').rglob('*.py'))+[ROOT/f'scripts/{name}.py' for name in ['train_joint_delivery_control','train_joint_tail_control','train_joint_parameterized_control']]
    fingerprints={str(f.relative_to(ROOT)):_hash_file(f) for f in files}
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_commit=source,prepared_source_commit=m['git_commit'],prepared_manifest_sha256=_hash_file(prepared),
        predictor_sha256=m['predictor_sha256'],primary_agents=primary,support_agents=supports,episodes=128,seeds=[7,17,27],eval_every=32,
        primary_panels=dict(base=.5,tight=.5),primary_checkpoint='fixed_final128_no_peak_selection',main_training_episodes=3072,support_training_episodes=768,total_training_episodes=3840,
        prediction_probe='all48_final_full_same_weights_on_minus_off',secondary='all30final models on frozen existing8 mobile_base scenes; full24forecastoff; zero training',
        source_fingerprints=fingerprints,formal_holdout='sealed_not_evaluated')
    old_cache=next((ROOT/'artifacts/joint_tail_control/cache_offload_drl').glob('*/manifest.json'))
    plan['legacy_cache_reference']=dict(manifest=str(old_cache),manifest_sha256=_hash_file(old_cache),training_added=0,scope='same128x3 historical frozen controller, exact same physics and data; retain strong old policy alongside new protected comparator')
    (root/'comparison_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    run([py,'scripts/evaluate_joint_delivery_support.py','prepare','--root',str(root)],root/'mobility_support_prepare.log')
    import torch,numpy
    (root/'runtime.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,platform=platform.platform(),cpu_count=os.cpu_count(),torch_threads_per_trainer=1),indent=2)+'\n')
    def start(command,log):
        log.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
        with log.open('w') as stream:return subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
    jobs={};extra={};audit=None;collector=None
    for name in names:
        folder=root/name;folder.mkdir()
        jobs[name]=start([py,'scripts/train_joint_delivery_control.py','--protocol',str(prepared),'--agent',name,'--seeds','7','17','27','--episodes','128','--eval_every','32','--output_root',str(folder)],folder/'command.log')
    reference=start([py,'scripts/audit_joint_delivery_cache_reference.py','--root',str(root)],root/'legacy_cache_reference_command.log')
    for tick in range(600):
        states=[];episodes=0;complete=0
        for name,job in jobs.items():
            paths=list((root/name).glob('*/manifest.json'));status='starting';seed=7;ep=0
            if paths:
                try:
                    manifest=json.loads(paths[0].read_text());status=manifest['status']
                    rows=(paths[0].parent/'training.jsonl').read_text().splitlines();episodes+=len(rows)
                    if rows:last=json.loads(rows[-1]);seed=last['seed'];ep=last['episode']
                    if job.poll()==0 and status=='complete':
                        complete+=1
                        if name not in extra:extra[name]=start([py,'scripts/evaluate_joint_delivery_support.py','evaluate','--root',str(root),'--agent',name],root/'mobility_support'/f'{name}_command.log')
                except (ValueError,FileNotFoundError):status='updating'
            states.append(dict(agent=name,status=status,seed=seed,episode=ep,exit_code=job.poll()))
        failures={n:j.poll() for n,j in {**jobs,**{f'support_{n}':j for n,j in extra.items()}}.items() if j.poll() not in (None,0)}
        if audit is None and complete==10:
            assert all(_hash_file(ROOT/name)==sha for name,sha in fingerprints.items()),'frozen producer source changed'
            audit=start([py,'scripts/audit_joint_tail_control.py',str(root),'--delivery'],root/'audit_command.log')
        if collector is None and len(extra)==10 and all(j.poll()==0 for j in extra.values()):
            collector=start([py,'scripts/evaluate_joint_delivery_support.py','collect','--root',str(root)],root/'mobility_support/collect_command.log')
        progress=dict(at=datetime.now(timezone.utc).isoformat(),episodes=episodes,completed=complete,jobs=states,errors=failures,
            support_launched=len(extra),support_completed=sum(j.poll()==0 for j in extra.values()),audit_exit=audit.poll() if audit else 'pending',support_audit_exit=collector.poll() if collector else 'pending',legacy_cache_exit=reference.poll())
        (root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps({k:v for k,v in progress.items() if k!='jobs'}),flush=True)
        if audit is not None and collector is not None and audit.poll() is not None and collector.poll() is not None and reference.poll() is not None:
            assert audit.returncode==collector.returncode==0 and reference.poll()==0 and not failures
            (root/'launcher_result.json').write_text(json.dumps(dict(status='complete',source_commit=source,train_episodes=episodes,groups=complete),indent=2)+'\n');return
        if failures and all(j.poll() is not None for j in jobs.values()):raise RuntimeError(failures)
        if audit and audit.poll() not in (None,0):raise RuntimeError('Primary audit failed; original logs preserved')
        if collector and collector.poll() not in (None,0):raise RuntimeError('Support audit failed; original logs preserved')
        time.sleep(45)
    raise TimeoutError('Inspect child processes and manifests before restarting; never blindly rerun training')


if __name__=='__main__':main()
