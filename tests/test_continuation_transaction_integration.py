from __future__ import annotations
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')


@pytest.mark.parametrize('fault,dispatches,success', [
    ('none', 1, True), ('publication_crash', 1, True), ('finalize_only', 1, True), ('exit75', 2, True), ('exit7', 1, False),
    ('missing', 1, False), ('provenance', 1, False), ('descriptor', 1, False)])
def test_actual_pinned_publication_and_failure_paths(tmp_path, fault, dispatches, success, record_property):
    fixture = tmp_path / ('synthetic_' + fault)
    spec = importlib.util.spec_from_file_location('fixture_sandbox', ROOT / 'scripts/continuation_fixture_sandbox.py')
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    result = subprocess.run(sandbox.sandbox_command([sys.executable, '-I', '-B',
        str(ROOT / 'scripts/validate_continuation_ledger_fixture.py'),
        '--fixture-root', str(fixture), '--fault', fault], fixture, SCIENCE),
        cwd=SCIENCE, env=sandbox.fixture_environment(fixture, SCIENCE),
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((fixture / 'report.json').read_text())
    assert report['kernel_boundary']['status'] == 'pass'
    assert (report['error'] is None) is success
    assert (report['validation_error'] is None) is success
    counts = report['monitor']['counts']
    assert counts['synthetic_child_dispatch_count'] == dispatches
    assert counts['observed_scientific_calls'] > 0
    for key in ('scientific_rollout_count', 'real_v16_dispatch_count', 'real_v16_write_count'):
        assert counts[key] == 0
        record_property(key, counts[key])
    if success:
        assert report['validation']['immutable_committed_cells'] == 2
        assert report['validation']['prefix_unchanged'] is True
        assert len(report['child_monitors']) == 1
        assert report['child_monitors'][0]['counts']['observed_scientific_calls'] > 0
    else:
        assert not list((fixture / 'synthetic_run/formal_ablation').glob('**/committed_marker.json'))
    if fault == 'none':
        assert len(report['completion_rejections']) == 2
        assert all(row['ledger_unchanged'] and row['extra_dispatches'] == 0 for row in report['completion_rejections'])
        assert len(report['negative_checks']) == 10
        assert all(row['status'] == 'rejected' for row in report['negative_checks'])
    if success:
        assert report['recovery']['repeated_start_dispatches'] == 0
    else:
        assert report['recovery']['failed_terminal_restart'] == 'rejected'
    if fault in {'publication_crash', 'finalize_only'}:
        assert report['recovery']['native_recovery'] is True
        assert report['recovery']['extra_child_dispatches'] == 0
    assert report['scientific_modules']
    assert all(row['path'].startswith(str(SCIENCE) + '/') for row in report['scientific_modules'])
    record_property('synthetic_child_dispatch_count', counts['synthetic_child_dispatch_count'])
    record_property('scientific_source_root', str(SCIENCE))
