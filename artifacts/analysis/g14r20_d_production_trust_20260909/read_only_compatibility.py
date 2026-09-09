import sys,os,pathlib,json
R=pathlib.Path('/private/tmp/ppo_mec_g14r20_d_production_trust');O=pathlib.Path('/private/tmp/g14r20_d_evidence')
sys.path.insert(0,str(R/'scripts'))
os.environ['PYTHONPATH']=str(pathlib.Path.cwd())
from continuation_executor.identity import verify_executor,read_json,canonical,digest,file_hash
from continuation_executor.authorization import validate_contract
from continuation_executor.monitoring import Monitor
from continuation_executor.scientific import QualifiedRun
monitor=Monitor()
i=read_json(O/'executor_identity.json');verify_executor(i,R)
p=read_json('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json');c=read_json(O/'unsigned_execution_contract.json')
validate_contract(c,p,i)
q=QualifiedRun(p,c)
assert digest(q.plans)==c['command_plan_sha256']
d=dict(status='read_only_compatible_not_authorized',execution_authorized=False,reconciliation=q.reconciliation,checkpoint_audit=q.checkpoints,registry_audit=q.registry_audit,active_resource_count=len(q.bundle['resource_ids']),plans=q.plans,origins=q.origins,environment=q.environment.runtime_audit,monitor=monitor.report())
# No report file opens under the monitor: stdout is captured by the parent.
print(json.dumps(d,allow_nan=False))
