"""Isolated synthetic publication → actual statistics/integrity/gate/completion.
No grant and no real run identity. Synthetic CSV values are not research results.
"""
import csv,json,hashlib,subprocess
from pathlib import Path
import pytest
from tests.test_evaluation_only_live_consumers import consumer, source_reference, ROOT
from src.evaluators.formal_cell_transaction import CellExecutionIdentity, FormalCellLedger, validate_producer_integrity_manifests
from scripts.run_typed_model_cache_formal_protocol import validate_complete_without_holdout_gate
from src.evaluators.typed_model_cache_formal_execution import FormalExecutionError


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')


def test_actual_statistics_integrity_false_gate_completion(consumer,record_property):
    root,source,contract=consumer
    protocol=json.loads(Path(source['protocol_path']).read_text())
    write(root/'non_formal_rehearsal.json',{'test_only':True,'formal_performance_evidence':False})
    ledger=FormalCellLedger(run_root=root,identity=CellExecutionIdentity(
        run_id=root.name,execution_commit=contract['executor_commit'],
        protocol_semantic_sha256=source['protocol_semantic_sha256'],
        resource_registry_semantic_sha256='fixture',environment_fingerprint='fixture',
        split_semantic_sha256='fixture',window_contract_semantic_sha256='fixture',
        catalog_fingerprint='fixture',runtime_identity='fixture',
        command_matrix_sha256=contract['command_plan_sha256']))
    for capacity in ['constrained_288mb','medium_576mb','relaxed_864mb']:
        begun=ledger.begin_cell(phase='formal_controller',coordinates={'capacity_label':capacity},
            command=['synthetic_fixture_only'],input_hash='fixture',committed_path=root/'formal_controller'/capacity)
        staging=Path(begun['record']['staging_path'])
        artifact=staging/'artifact';producer=artifact/'benchmark'/'synthetic'
        write(producer/'aggregate_summary.json',{'test_only':True,'output_paths':{'aggregate':str(producer/'aggregate_summary.json')}})
        write(producer/'run_manifest.json',{'test_only':True,'output_paths':{'aggregate':str(producer/'aggregate_summary.json')}})
        metrics=protocol['endpoints']['primary']
        fields=['agent_name','seed','window_id','workflow_id','source_segment_run_id',*metrics]
        with (producer/'benchmark_rows.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for agent in ['sa_ghmappo','ppo']:
                for window in range(2):
                    writer.writerow(dict(agent_name=agent,seed=7,window_id=f'fixture_{window}',workflow_id='synthetic',source_segment_run_id=capacity,**{m:1+window/10 for m in metrics}))
        files=[{'path':str(p.relative_to(producer)),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(producer.iterdir())]
        write(producer/'artifact_integrity_manifest.json',{'integrity_manifest_version':'1.0.0','files':files})
        event=ledger.commit_cell(begun['cell_id'],validated_artifact_root=artifact)
        assert validate_producer_integrity_manifests(Path(event['committed_path']),require_manifest=True)['status']=='pass'
        ledger.verify_committed(begun['cell_id'])
    command=contract['command_plans']['formal_statistics']['commands'][0]+['--non-formal-rehearsal','--rehearsal-baseline-agent','ppo']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    record_property('statistics_command',json.dumps(command));record_property('statistics_stdout',result.stdout);record_property('statistics_stderr',result.stderr)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['row_artifact_count']==3
    statistics=json.loads((root/'statistics/paired_statistics.json').read_text())
    assert statistics['formal_performance_evidence'] is False
    # No cache/support cells: the actual gate must produce passed=false, never completion.
    command=contract['command_plans']['formal_gate']['commands'][0]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    record_property('gate_command',json.dumps(command));record_property('gate_stdout',result.stdout);record_property('gate_stderr',result.stderr)
    assert result.returncode==2,result.stderr
    assert json.loads((root/'formal_gate.json').read_text())['passed'] is False
    assert (root/'artifact_integrity_manifest.json').is_file()
    with pytest.raises(FormalExecutionError):validate_complete_without_holdout_gate(root,protocol)
    (root/'formal_gate.json').unlink()
    with pytest.raises(FormalExecutionError):validate_complete_without_holdout_gate(root,protocol)
    assert not (root/'phase_state.jsonl').exists()
    record_property('synthetic_only',True)
    record_property('real_rollout_training_grant_counts','0/0/0')
