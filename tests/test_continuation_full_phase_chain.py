from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SCIENCE=Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')


def test_actual_eight_phase_chain(tmp_path, record_property):
    spec=importlib.util.spec_from_file_location('fixture_sandbox',ROOT/'scripts/continuation_fixture_sandbox.py')
    sandbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(sandbox)
    fixture=tmp_path/'synthetic_full_chain'
    result=subprocess.run(sandbox.sandbox_command([sys.executable,'-I','-B',
        str(ROOT/'scripts/validate_continuation_phase_chain.py'),'--fixture-root',str(fixture)],fixture,SCIENCE),
        cwd=SCIENCE,env=sandbox.fixture_environment(fixture,SCIENCE),text=True,capture_output=True,check=False)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    report=json.loads((fixture/'phase_chain_report.json').read_text())
    assert report['error'] is None
    assert len(report['results'])==8
    assert report['prefix_unchanged'] is True
    counts=report['monitor']['counts']
    assert counts['synthetic_child_dispatch_count']==24
    for key in ['scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count']:
        assert counts[key]==0
        record_property(key,counts[key])
    gate=json.loads((fixture/'synthetic_run/formal_gate.json').read_text())
    assert gate['passed'] is True and gate['execution_mode']=='non_formal_rehearsal'
    assert gate['observed_counts']['frozen_checkpoints']==150
    assert gate['observed_counts']['support_settings']==11
    assert gate['observed_counts']['primary_comparison_rows']==6
    record_property('parent_synthetic_child_dispatch_count',counts['synthetic_child_dispatch_count'])
    record_property('completed_continuation_phases',8)

    assert len(report['child_observations'])==24
    assert report['aggregate_process_counts']['synthetic_child_dispatch_count']==25
    for name in ['scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count']:
        assert report['aggregate_process_counts'][name]==0
    assert all(row['dispatch_authority']['domain']=='synthetic-only' for row in report['child_observations'])
