import sys
from pathlib import Path
sys.path.insert(0,'/private/tmp/ppo_mec_g14r20_b_62f432c/scripts')
from continuation_executor.identity import read_json,canonical,digest,file_hash
from continuation_executor.scientific import load_native
from continuation_executor.planning import phase_plan
from continuation_executor import PHASES
from continuation_executor.authorization import validate_contract
pfile=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json')
p=read_json(pfile);run=Path(p['run_root'])
e=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
identity=read_json(e/'executor_identity.json')
n=load_native(str(Path.cwd()),p['execution_commit'])
context=read_json(run/'resolved_execution_context.json');binding=read_json(run/'formal_training_execution_binding.json')
protocol=read_json(context['runtime_location']['protocol_path'])
registry=read_json(run/'generated_checkpoint_resource_registry.json')
plans={phase:phase_plan(phase,protocol,context,binding,file_hash(run/'resolved_execution_context.json'),registry['registry_canonical_sha256'],n['execution'].expand_command_plan) for phase in PHASES}
contract=dict(version='1.0.0',domain='production',proposal_sha256=digest(p),proposal_file_sha256=file_hash(pfile),executor_identity_sha256=digest(identity),
 run_id=p['run_id'],run_root=p['run_root'],phases=list(PHASES),holdout_capability=False,prefixes=p['ledgers'],
 immutable_files=[{k:r[k] for k in ('path','sha256','size_bytes')} for r in p['evidence']],fixture_root=None,
 expires_at='2026-09-10T00:00:00+00:00',revocation_id='g14r20-b-unsigned-request',recovery_owner_sha256=None,recovery_quiescence=None,
 coordination_root=str(run.parent/'.continuation_locks'),command_plan_sha256=digest(plans))
validate_contract(contract,p,identity)
request=dict(version='1.0.0',state='pending_independent_review',execution_authorized=False,contract_sha256=digest(contract),executor_identity_sha256=digest(identity),
 launch_approval=dict(state='unavailable',reference_sha256=None),release_attestation=dict(state='unavailable',reference_sha256=None),
 continuation_approval=dict(state='pending',reference_sha256=None),boundary='Unsigned request only; not approval; no production trust installed.')
for filename,value in [('unsigned_execution_contract.json',contract),('approval_request.json',request),('frozen_command_plans.json',plans)]:
 with (e/filename).open('xb') as f:f.write(canonical(value)+b'\n')
print('unsigned request prepared',len(plans),'phases',len(contract['immutable_files']),'immutable identities')
