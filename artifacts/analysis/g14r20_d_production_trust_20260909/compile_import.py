import sys,pathlib,importlib,json,subprocess,hashlib
R=pathlib.Path('/private/tmp/ppo_mec_g14r20_d_production_trust');O=pathlib.Path('/private/tmp/g14r20_d_evidence')
sys.path.insert(0,str(R/'scripts'))
from continuation_executor.identity import file_hash,canonical,verify_executor,read_json
verify_executor(read_json(O/'executor_identity.json'),R)
paths=subprocess.check_output(['git','-C',str(R),'ls-files','*.py'],text=True).splitlines()
for p in paths:compile((R/p).read_text(encoding='utf-8-sig'),str(R/p),'exec')
for p in (R/'scripts/continuation_executor').glob('*.py'):
 importlib.import_module('continuation_executor.'+p.stem)
import cryptography
report=dict(status='pass',compiled_python_count=len(paths),compiled_files=[dict(path=p,sha256=file_hash(R/p)) for p in paths],cryptography_version=cryptography.__version__,cryptography_path=cryptography.__file__)
(O/'compile_import_report.json').write_bytes(canonical(report)+b'\n');print('compile/import pass',len(paths))
d=pathlib.Path('/private/tmp/g14r20_d_dependencies')
files=[dict(path=str(p.relative_to(d)),sha256=file_hash(p),size_bytes=p.stat().st_size) for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
(O/'authorization_dependency_identity.json').write_bytes(canonical(dict(root=str(d),files=files,sha256=hashlib.sha256(canonical(files)).hexdigest()))+b'\n')
