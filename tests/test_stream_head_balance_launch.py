"""Synthetic launcher failure/claim/budget witnesses; no policy or environment."""
import json
from pathlib import Path
import sys
import time
import types
import pytest

from scripts import launch_stream_head_balance_study as launch


def plan(tmp_path):
    return dict(root=str(tmp_path/'new_run'), source_commit='frozen', workers=2,
        maximum_scientific_launches=1, replay_episodes=launch.REPLAYS,
        global_deadline_unix=time.time()+3600, maximum_training_frames=230400,
        maximum_evaluation_frames=115200, maximum_replay_frames=1440,
        maximum_total_physical_frames=347040, maximum_wall_seconds=86400,
        maximum_disk_bytes=3221225472,
        outputs={launch.METHOD: {str(s): str(tmp_path/f'cell{s}') for s in launch.SEEDS}})


def test_clean_freeze_and_prospective_envelope(tmp_path, monkeypatch):
    candidate = types.ModuleType('scripts.run_stream_head_balance_study')
    calls = []
    candidate.validate_plan = lambda *a: calls.append(a[2])
    monkeypatch.setitem(sys.modules, candidate.__name__, candidate)
    monkeypatch.setattr(launch.subprocess, 'check_output',
                        lambda cmd, **kw: '' if cmd[1] == 'status' else 'frozen\n')
    data = plan(tmp_path)
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(data))
    launch.validate_campaign(data, path, launch.file_hash(path))
    assert calls == list(launch.SEEDS)
    data['maximum_total_physical_frames'] += 1
    with pytest.raises(ValueError, match='envelope'):
        launch.validate_campaign(data, path, launch.file_hash(path))


def test_tampered_plan_rejected_before_cells(tmp_path):
    path = tmp_path/'plan.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='SHA'):
        launch.validate_campaign({}, path, '0'*64)


def test_deadline_and_disk_reserve_are_hard(tmp_path):
    data = plan(tmp_path)
    Path(data['root']).mkdir()
    data['global_deadline_unix'] = 0
    with pytest.raises(RuntimeError, match='deadline'):
        launch.guard(data)
    data['global_deadline_unix'] = time.time()+3600
    data['maximum_disk_bytes'] = 8*1024**2
    (Path(data['root'])/'bytes').write_bytes(b'x')
    with pytest.raises(RuntimeError, match='disk'):
        launch.guard(data)


def test_failure_preserved_no_relaunch_no_collection(tmp_path, monkeypatch):
    data = plan(tmp_path)
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(launch, 'validate_campaign', lambda *a: None)
    commands = []
    class Failed:
        pid = 999999
        returncode = 9
        def __init__(self, cmd, **kw):
            commands.append(cmd)
        def poll(self):
            return self.returncode
    monkeypatch.setattr(launch.subprocess, 'Popen', Failed)
    with pytest.raises(RuntimeError, match='no retry'):
        launch.run(path, launch.file_hash(path))
    root = Path(data['root'])
    failure = json.loads((root/'failure.json').read_text())
    assert failure['status'] == 'failed_preserved'
    assert not (root/'complete.json').exists()
    assert len(commands) == 2
    assert all('collect_stream_head_balance_study.py' not in ' '.join(c) for c in commands)
    assert (tmp_path/'head_balance_execution_claim.json').exists()
    with pytest.raises(FileExistsError):
        launch.run(path, launch.file_hash(path))
    assert len(commands) == 2


def test_success_exit_without_collection_receipt_is_failure(tmp_path, monkeypatch):
    data = plan(tmp_path)
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(launch, 'validate_campaign', lambda *a: None)
    monkeypatch.setattr(launch.time, 'sleep', lambda *a: None)
    commands = []
    class Empty:
        pid = 999999
        returncode = 0
        def __init__(self, cmd, **kw):
            commands.append(cmd)
        def poll(self):
            return 0
    monkeypatch.setattr(launch.subprocess, 'Popen', Empty)
    with pytest.raises(ValueError, match='omitted complete'):
        launch.run(path, launch.file_hash(path))
    assert len(commands) == 6
    assert json.loads((Path(data['root'])/'failure.json').read_text())['status'] == 'failed_preserved'
