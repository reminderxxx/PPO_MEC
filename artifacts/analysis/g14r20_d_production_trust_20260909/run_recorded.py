import sys,subprocess,pathlib,json,os,datetime,time
R=pathlib.Path('/private/tmp/ppo_mec_g14r20_d_production_trust'); O=pathlib.Path('/private/tmp/g14r20_d_evidence')
name=sys.argv[1];cwd=sys.argv[2];cmd=sys.argv[3:]
def git(*a):return subprocess.check_output(['git','-C',str(R),*a],text=True).strip()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH='/private/tmp/g14r20_d_dependencies')
if name.startswith('synthetic'):env['PYTHONPATH']=cwd
d=dict(name=name,command=cmd,cwd=cwd,environment={k:env[k] for k in ['PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','PYTHONPATH']},started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),executor_commit_before=git('rev-parse','HEAD'),executor_tree=git('rev-parse','HEAD^{tree}'),executor_status_before=git('status','--porcelain','--untracked-files=all'))
assert not d['executor_status_before'],d
(O/(name+'_command.json')).write_text(json.dumps(d,indent=2)+'\n')
t=time.monotonic()
with (O/(name+'_stdout.log')).open('w') as out,(O/(name+'_stderr.log')).open('w') as err:
 p=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err)
d.update(exit_code=p.returncode,elapsed_seconds=time.monotonic()-t,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),executor_commit_after=git('rev-parse','HEAD'),executor_status_after=git('status','--porcelain','--untracked-files=all'))
(O/(name+'_result.json')).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));sys.exit(p.returncode)
