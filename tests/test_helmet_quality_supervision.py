"""Two bounded, non-model delta cases for QPROD01/02; run once."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
P=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_evidence_strengthening_20261029_v1/helmet_quality_pilot_20261030_v1/producer_delta_software')
spec=importlib.util.spec_from_file_location('qdelta',ROOT/'scripts/run_helmet_quality_pilot.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def test_alarm_pg_cleanup_durable_control(tmp_path):
    P=tmp_path
    previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('fixture outer deadline')))
    signal.setitimer(signal.ITIMER_REAL,0.2)
    try:
        control=q.supervise([sys.executable,'-c',"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"],P/'child.log',2,P/'control.json')
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    assert control['terminated_group'] and 'fixture outer deadline' in control['exception']
    assert control['returncode'] is not None and control['finished']-control['started']<4
    assert json.loads((P/'control.json').read_text())==control
    with pytest.raises(ProcessLookupError):__import__('os').killpg(control['pid'],0)


def test_shared_budget_and_stdout_reservation(tmp_path):
    folder=tmp_path/'budget';folder.mkdir(exist_ok=False)
    (folder/'prior.json').write_bytes(b'x'*200)
    cap=q.Budget([folder],artifact_limit=4096,log_limit=2048,reserve=512)
    assert cap.check()==(200,200)
    assert cap.allocate_log(folder/'worker.log')==1336
    with pytest.raises(ValueError,match='hard cap'):cap.check(extra=1,is_log=True)
    cap.lease_path=None;cap.lease_bytes=0
    with pytest.raises(ValueError,match='hard cap'):cap.check(extra=1337,is_log=True)
    (folder/'raw.bin').write_bytes(b'x'*3500)
    with pytest.raises(ValueError,match='artifact'):cap.check()
