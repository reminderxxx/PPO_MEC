from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCIENCE=Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')


@pytest.mark.parametrize('fault',['expire','revoke'])
def test_inflight_native_phase_finishes_but_next_phase_denied(tmp_path,fault):
    spec=importlib.util.spec_from_file_location('sandbox',ROOT/'scripts/continuation_fixture_sandbox.py')
    sandbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(sandbox)
    fixture=tmp_path/'synthetic_phase_lease'
    result=subprocess.run(sandbox.sandbox_command([sys.executable,'-I','-B',
        str(ROOT/'scripts/validate_continuation_phase_chain.py'),'--fixture-root',str(fixture),
        '--lease-fault',fault],fixture,SCIENCE),cwd=SCIENCE,
        env=sandbox.fixture_environment(fixture,SCIENCE),text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((fixture/'phase_chain_report.json').read_text())
    assert report['expected_lease_denial'] and report['lease_triggered']
    assert len(report['results'])==1 and report['results'][0]['phase']=='formal_cache_policy'
    rows=[json.loads(line) for line in (fixture/'synthetic_run/phase_state.jsonl').read_text().splitlines()]
    assert rows[-1]['phase']=='formal_cache_policy' and rows[-1]['status']=='completed'
    assert all(row['phase']!='formal_controller' for row in rows)
    assert report['prefix_unchanged']
    assert report['aggregate_process_counts']['synthetic_child_dispatch_count']==3
    for name in ['scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count']:
        assert report['aggregate_process_counts'][name]==0
