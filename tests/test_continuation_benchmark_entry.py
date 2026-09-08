from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SCIENCE=Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')


def test_actual_benchmark_main_native_gates_before_rollout(tmp_path):
    spec=importlib.util.spec_from_file_location('external_sandbox',ROOT/'scripts/continuation_fixture_sandbox.py')
    sandbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(sandbox)
    fixture=tmp_path/'synthetic_benchmark'
    result=subprocess.run(sandbox.sandbox_command([sys.executable,'-I','-B',
        str(ROOT/'scripts/validate_continuation_benchmark_fixture.py'),'--fixture-root',str(fixture)],fixture,SCIENCE),
        cwd=SCIENCE,env=sandbox.fixture_environment(fixture,SCIENCE),text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((fixture/'benchmark_report.json').read_text())
    assert report['boundary_reached'] and report['error'] is None
    assert len(report['cases'])==30
    assert len({(case['agent'],case['capacity']) for case in report['cases']})==30
    assert all(case['boundary_reached'] for case in report['cases'])
    calls=report['benchmark_call_counts']
    for name in ['main','load_seed_checkpoint_manifest','load_checkpoint_provenance_manifest',
                 'resolve_argument_resources','resolve_generated_checkpoint_arguments',
                 'load_and_validate_manifest','enforce_benchmark_args','validate_benchmark_checkpoint_gate']:
        assert any(key.endswith('.'+name) and count>0 for key,count in calls.items()),name
    counts=report['monitor']['counts']
    for name in ['scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count']:
        assert counts[name]==0
    assert report['scientific_modules']
    assert all(row['path'].startswith(str(SCIENCE)+'/') for row in report['scientific_modules'])
