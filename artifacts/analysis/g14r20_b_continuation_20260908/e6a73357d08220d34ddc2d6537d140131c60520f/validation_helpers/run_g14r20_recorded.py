import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
job_path=Path(sys.argv[1]);job=json.loads(job_path.read_text())
base=job_path.parent;name=job['name'];start=time.monotonic()
record=dict(job,started_at=datetime.now(timezone.utc).isoformat())
def git(*args):return subprocess.check_output(['git','-C',job['executor_root'],*args],text=True,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')).strip()
record['executor_commit_before']=git('rev-parse','HEAD');record['executor_status_before']=git('status','--porcelain','--untracked-files=all')
with (base/(name+'_stdout.log')).open('xb') as out,(base/(name+'_stderr.log')).open('xb') as err:
 result=subprocess.run(job['argv'],cwd=job['cwd'],env=dict(os.environ,**job.get('env',{})),stdout=out,stderr=err)
record.update(exit_code=result.returncode,duration_seconds=time.monotonic()-start,ended_at=datetime.now(timezone.utc).isoformat(),executor_commit_after=git('rev-parse','HEAD'),executor_status_after=git('status','--porcelain','--untracked-files=all'))
with (base/(name+'_command_result.json')).open('x') as stream:json.dump(record,stream,sort_keys=True,allow_nan=False,indent=2)
print(json.dumps(dict(name=name,exit_code=result.returncode,duration_seconds=record['duration_seconds'])))
sys.exit(result.returncode)
