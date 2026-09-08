from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCIENCE=Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')
MAIN=Path('/Users/howen/Projects/PPO_MEC')


@pytest.mark.parametrize('case,reason',[
    ('wrong_cwd','cwd drift'),('wrong_pythonpath','PYTHONPATH drift'),
    ('main_src','fresh interpreter'),('external_src','fresh interpreter'),
])
def test_actual_scientific_adapter_refuses_environment_or_loaded_source_contamination(tmp_path,case,reason):
    source=SCIENCE/'scripts/run_typed_model_cache_formal_protocol.py'
    code="""import importlib.util,sys,json
from pathlib import Path
root,science,source_hash,contamination=sys.argv[1:]
if contamination:
    sys.path.insert(0,contamination)
    import src
    print(json.dumps({'contaminating_module':src.__file__}))
spec=importlib.util.spec_from_file_location('external_adapter',Path(root)/'scripts/continuation_legacy_adapter.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
    module.load_scientific_modules(science,source_hash,expected_commit='a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')
except ValueError as exc:
    print(json.dumps({'rejected':str(exc)}));raise SystemExit(2)
raise SystemExit('unexpected admission')
"""
    contamination=''
    if case=='main_src':contamination=str(MAIN)
    if case=='external_src':
        shadow=tmp_path/'external';(shadow/'src').mkdir(parents=True)
        (shadow/'src/__init__.py').write_text('TEST_ONLY_SHADOW = True\n')
        contamination=str(shadow)
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT),str(SCIENCE),
        hashlib.sha256(source.read_bytes()).hexdigest(),contamination],
        cwd=tmp_path if case=='wrong_cwd' else SCIENCE,
        env=dict(os.environ,PYTHONPATH=str(tmp_path) if case=='wrong_pythonpath' else str(SCIENCE)),
        text=True,capture_output=True)
    assert result.returncode==2,result.stdout+result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert reason in rows[-1]['rejected']
    if contamination:assert rows[0]['contaminating_module'].startswith(contamination+'/')
