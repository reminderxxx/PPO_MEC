import sys,importlib.util,json
from pathlib import Path
from datetime import datetime,timezone
root=Path('/private/tmp/ppo_mec_g14r20_b_62f432c');sys.path.insert(0,str(root/'scripts'))
from continuation_executor.identity import read_json,file_hash,canonical
base=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908');out=base/'e6a73357d08220d34ddc2d6537d140131c60520f'
spec=importlib.util.spec_from_file_location('protected_a_validator',root/'src/runtime/fixed_commit_continuation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=read_json('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json')
inv=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/protected_before.json.gz')
start=read_json(base/'startup_identity.json')
assert file_hash(inv)==start['protected_inventory_file_sha256']
report=m.validate_protected_inventory(inv,immutable_roots=[p['run_root'],p['worktree_root']])
report['payloads']=m.validate_committed_payloads(p['run_root'],inv)
report['checked_at']=datetime.now(timezone.utc).isoformat()
(out/'protected_inventory_verified.json').write_bytes(canonical(report)+b'\n')
report['user_files_start']=start['user_files']
print('user file schema',type(start['user_files']).__name__,flush=True)
rows=start['user_files']; after={}
if isinstance(rows,dict):
 for path,expected in rows.items():
  observed=file_hash(path if Path(path).is_absolute() else Path('/Users/howen/Projects/PPO_MEC')/path)
  assert observed==(expected['sha256'] if isinstance(expected,dict) else expected)
  after[path]=observed
else:
 for row in rows:
  observed=file_hash(row['path']);assert observed==row['before_sha256'];after[row['path']]=observed
report['user_files_end']=after
assert len(after)==7
(out/'protected_final.json').write_bytes(canonical(report)+b'\n')
print(json.dumps({'status':'pass','entries':report['verified_entries'],'bytes':report['verified_bytes'],'user_files':len(after)}))
