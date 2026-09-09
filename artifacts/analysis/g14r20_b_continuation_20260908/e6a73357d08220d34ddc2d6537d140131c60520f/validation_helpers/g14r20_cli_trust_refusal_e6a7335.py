import sys,runpy,io,json,hmac,hashlib
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from datetime import datetime,timedelta,timezone
root=Path('/private/tmp/ppo_mec_g14r20_b_62f432c');sys.path.insert(0,str(root/'scripts'))
from continuation_executor.identity import read_json,canonical,digest,file_hash
from continuation_executor.authorization import approval_message
from continuation_executor.monitoring import Monitor
b=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f');fixture=b/'synthetic_acceptance_01';out=b/'cli_negative_test_only';out.mkdir(exist_ok=True)
p=read_json(fixture/'proposal_test_only.json');contract=read_json(fixture/'contract_test_only.json');authority=read_json(fixture/'authority_test_only.json')
(out/'proposal.json').write_bytes(canonical(p)+b'\n')
cases=[]
for name in ('synthetic_domain','forged_production_domain'):
 c=dict(contract)
 c["expires_at"]=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
 if name=='forged_production_domain':c.update(domain='production',fixture_root=None)
 evidence={key:dict(state='test_only' if c['domain']=='synthetic' else 'independently_verified',reference_sha256=digest('counterfeit negative fixture')) for key in ('launch_approval','release_attestation','continuation_approval')}
 message=approval_message(c,evidence);approval=dict(message=message,signer_id=authority['signer_id'],signature=hmac.new(bytes.fromhex(authority['key_hex']),canonical(message),hashlib.sha256).hexdigest())
 (out/(name+'_contract.json')).write_bytes(canonical(c));(out/(name+'_counterfeit_test_only.json')).write_bytes(canonical(approval));cases.append(name)
monitor=Monitor(None);reports=[]
for name in cases:
 argv=[str(root/'scripts/execute_fixed_commit_continuation.py'),'--proposal',str(out/'proposal.json'),'--contract',str(out/(name+'_contract.json')),'--executor-identity',str(b/'executor_identity.json'),'--approval',str(out/(name+'_counterfeit_test_only.json')),'--phase','formal_cache_policy','--check','execute']
 old=sys.argv;sys.argv=argv;stdout=io.StringIO();stderr=io.StringIO();code=0
 try:
  with redirect_stdout(stdout),redirect_stderr(stderr):
   try:runpy.run_path(argv[0],run_name='__main__')
   except SystemExit as exc:code=exc.code
 finally:sys.argv=old
 result=json.loads(stdout.getvalue());expected='production entry refuses synthetic trust' if name=='synthetic_domain' else 'independent production trust unavailable'
 print(name,code,result, file=sys.stderr)
 assert code==2 and expected in result['reason'] and result['execution_authorized'] is False
 reports.append(dict(case=name,status='pass',argv=argv,result=result,exit_code=code))
m=monitor.report();assert m['observed_write_events']==0 and m['actual_function_calls']['scripts.execute_fixed_commit_continuation.main']==2
assert m['synthetic_child_dispatch_count']==m['scientific_rollout_count']==m['real_v16_dispatch_count']==m['real_v16_write_count']==0
(b/'cli_trust_refusal_report.json').write_bytes(canonical(dict(status='pass',negative_test_only=True,real_approval_created=False,cases=reports,monitor=m))+b'\n')
print('actual production CLI rejected both test-trust cases before writes/dispatch')
