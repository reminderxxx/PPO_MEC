import sys,runpy,json
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
sys.path.insert(0,'/private/tmp/ppo_mec_g14r20_b_62f432c/scripts')
from continuation_executor.monitoring import Monitor
from continuation_executor.identity import canonical
base=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
entry='/private/tmp/ppo_mec_g14r20_b_62f432c/scripts/execute_fixed_commit_continuation.py'
sys.argv=[entry,'--proposal','/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json','--contract',str(base/'unsigned_execution_contract.json'),'--executor-identity',str(base/'executor_identity.json'),'--phase','formal_cache_policy','--check','compatibility']
(base/'readonly_cli_command.json').write_bytes(canonical(dict(argv=sys.argv,cwd=str(Path.cwd()),probe_python_flags=['-I','-B'],frozen_child_argv_modified=False))+b'\n')
monitor=Monitor(None)
code=0
with (base/'readonly_compatibility.json').open('w') as out,(base/'readonly_stderr.log').open('w') as err,redirect_stdout(out),redirect_stderr(err):
 try:runpy.run_path(entry,run_name='__main__')
 except SystemExit as exc:code=exc.code
report=monitor.report();report['cli_exit_code']=code
(base/'readonly_monitor.json').write_bytes(canonical(report)+b'\n')
print(json.dumps(dict(exit_code=code,report=str(base/'readonly_compatibility.json'))))
