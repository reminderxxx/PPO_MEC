from pathlib import Path
import json,shutil,hashlib
b=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
assert json.loads((b/'acceptance_summary.json').read_text())['status']=='technical_acceptance_pass'
fixture=b/'synthetic_acceptance_01';out=b/'serialized_test_only';out.mkdir(exist_ok=True)
for name in ('proposal_test_only.json','contract_test_only.json','approval_test_only.json','authority_test_only.json','command_plan_test_only.json','executor_identity.json'):
 shutil.copyfile(fixture/name,out/name)
r=json.loads((b.parent/'implementation_supersession.json').read_text());r['replacement_verification']='technical_acceptance_pass';(b.parent/'implementation_supersession.json').write_text(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
coverage=[
 ('1_baseline_immutability',['executor_identity.json','protected_final.json','../startup_identity.json']),
 ('2_independent_objects_authorization',['unsigned_execution_contract.json','approval_request.json','cli_trust_refusal_report.json','serialized_test_only/contract_test_only.json']),
 ('3_fixed_executor_original_science',['compile_import_report.json','executor_verification.json','compatibility_details.json']),
 ('4_eight_phase_commands_identities',['compatibility_details.json','frozen_command_plans.json','synthetic_acceptance_report.json']),
 ('5_isolation_actual_counters',['synthetic_acceptance_report.json','fault_and_boundary_cases.json','cli_trust_refusal_report.json']),
 ('6_three_layers',['targeted_pytest.xml','synthetic_acceptance_report.json','staging_refusal_report.json']),
 ('7_recovery_single_writer',['fault_and_boundary_cases.json','targeted_pytest.xml','acceptance_summary.json']),
 ('8_environment_source',['compile_import_report.json','compatibility_details.json','synthetic_acceptance_report.json','targeted_pytest.xml']),
 ('9_real_read_only',['readonly_compatibility.json','readonly_monitor.json','protected_final.json','approval_request.json']),
 ('10_validation_delivery',['full_pytest.xml','targeted_pytest.xml','smoke_command_result.json','artifact_integrity.json','versioned_evidence_manifest.json'])]
(b/'requirements_evidence_index.json').write_text(json.dumps(dict(status='technical_acceptance_pass',implementation_commit=b.name,coverage=[dict(requirement=k,evidence=v) for k,v in coverage],production_execution_authorized=False,notes=['Git identities and remote synchronization are recorded in delivery_record.json after the separate evidence commit.','Synthetic authorization files contain test-only trust scoped to the isolated fixture, not production approval.']),sort_keys=True,allow_nan=False,indent=2)+'\n')
for name in ('g14r20_validate_artifacts_e6a7335.py','g14r20_finish_evidence.py','g14r20_export_evidence.py','g14r20_finalize_docs.py'):
 shutil.copyfile(Path('/private/tmp')/name,b/'validation_helpers'/name)
print('serialized test-only evidence and requirements index prepared')
