import sys,importlib.util
from pathlib import Path
root=Path('/private/tmp/ppo_mec_g14r20_b_62f432c');sys.path.insert(0,str(root/'scripts'))
from continuation_executor.scientific import load_native,loaded_origins
from continuation_executor.monitoring import Monitor
from continuation_executor.identity import canonical,file_hash
b=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f');fixture=b/'synthetic_staging_negative';fixture.mkdir()
n=load_native(str(Path.cwd()),'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')
spec=importlib.util.spec_from_file_location('frozen_test_fixture',root/'tests/test_cell_artifact_publication_v26.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
monitor=Monitor(fixture);run=fixture/'synthetic_staging';fields=m.identity('synthetic_staging').to_dict();fields['execution_commit']='a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d'
ledger=n['cell'].FormalCellLedger(run_root=run,identity=n['cell'].CellExecutionIdentity(**fields));begun,child,descriptor=m.prepare_support_payload(ledger)
wrong=fixture/'wrong_staging_root';wrong.mkdir();before={str(p):file_hash(p) for p in fixture.rglob('*') if p.is_file()}
try:n['cell'].resolve_child_output_descriptor(descriptor,output_root=wrong,expected_cell_id=begun['cell_id'],expected_phase='formal_support',expected_setting_id='setting-a')
except n['cell'].CellTransactionError as exc:reason=str(exc)
else:raise AssertionError('mismatched staging root accepted')
assert before=={str(p):file_hash(p) for p in fixture.rglob('*') if p.is_file()}
report=dict(status='pass',case='staging_output_root_mismatch',reason=reason,scientific_origins=loaded_origins(Path.cwd()),monitor=monitor.report(),scope='native descriptor boundary used by B; no child or public main')
assert report['monitor']['actual_function_calls']['src.evaluators.formal_cell_transaction.resolve_child_output_descriptor']==1
(fixture/'report.json').write_bytes(canonical(report)+b'\n');print(reason)
