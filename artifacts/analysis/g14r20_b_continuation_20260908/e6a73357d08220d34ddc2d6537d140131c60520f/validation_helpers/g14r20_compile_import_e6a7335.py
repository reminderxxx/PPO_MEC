import sys,importlib,importlib.util
from pathlib import Path
root=Path('/private/tmp/ppo_mec_g14r20_b_62f432c');sys.path.insert(0,str(root/'scripts'))
from continuation_executor.identity import canonical,verify_executor,read_json,file_hash
from continuation_executor.scientific import load_native,loaded_origins
base=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
identity=read_json(base/'executor_identity.json');verify_executor(identity,root)
n=load_native(str(Path.cwd()),'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')
files=list((root/'scripts/continuation_executor').glob('*.py'))+[root/'scripts/execute_fixed_commit_continuation.py',root/'scripts/run_fixed_commit_continuation_acceptance.py',root/'tests/continuation_native_driver.py',root/'tests/test_continuation_executor_boundaries.py']
for p in files:compile(p.read_text(),str(p),'exec')
for p in (root/'scripts/continuation_executor').glob('*.py'):
 if p.stem!='__init__':importlib.import_module('continuation_executor.'+p.stem)
for filename in ('execute_fixed_commit_continuation.py','run_fixed_commit_continuation_acceptance.py'):
 spec=importlib.util.spec_from_file_location('verify_'+filename[:-3],root/'scripts'/filename);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
r=dict(status='pass',compiled_files=[dict(path=str(p),sha256=file_hash(p)) for p in files],scientific_origins=loaded_origins(Path.cwd()),actual_parent_environment=n['process_environment'])
(base/'compile_import_report.json').write_bytes(canonical(r)+b'\n');print('compile/import pass',len(files))
