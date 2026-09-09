import sys,pathlib
R=pathlib.Path('/private/tmp/ppo_mec_g14r20_d_production_trust');sys.path.insert(0,str(R/'scripts'))
from continuation_executor.identity import git,file_hash,canonical,digest,verify_executor,read_json
O=pathlib.Path('/private/tmp/g14r20_d_evidence')
files=sorted(list((R/'scripts/continuation_executor').glob('*.py'))+[R/'scripts/execute_fixed_commit_continuation.py',R/'scripts/run_fixed_commit_continuation_acceptance.py',R/'src/runtime/fixed_commit_continuation.py'])
i=dict(version='1.0.0',commit=git(R,'rev-parse','HEAD'),git_tree=git(R,'rev-parse','HEAD^{tree}'),files=[dict(path=str(p.relative_to(R)),sha256=file_hash(p)) for p in files]);(O/'executor_identity.json').write_bytes(canonical(i)+b'\n');(O/'executor_verification.json').write_bytes(canonical(verify_executor(i,R))+b'\n')
B=pathlib.Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
c=read_json(B/'unsigned_execution_contract.json');c.update(version='2.0.0',executor_identity_sha256=digest(i),scientific_commit='a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d',release_identity='UNAVAILABLE_NOT_ISSUED',trust_installation_id='UNINSTALLED_NOT_ISSUED')
(O/'unsigned_execution_contract.json').write_bytes(canonical(c)+b'\n');(O/'frozen_command_plans.json').write_bytes((B/'frozen_command_plans.json').read_bytes())
assert digest(read_json(O/'frozen_command_plans.json')) == c['command_plan_sha256']
print(i['commit'],i['git_tree'],len(files),digest(i),digest(c))
