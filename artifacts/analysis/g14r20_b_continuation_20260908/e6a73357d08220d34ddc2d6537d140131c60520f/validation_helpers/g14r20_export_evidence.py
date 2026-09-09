from pathlib import Path
import json,hashlib,shutil
commit='e6a73357d08220d34ddc2d6537d140131c60520f'
main=Path('/Users/howen/Projects/PPO_MEC');dev=Path('/private/tmp/ppo_mec_g14r20_b_62f432c')
rel=Path('artifacts/analysis/g14r20_b_continuation_20260908');b=main/rel/commit;target=dev/rel/commit
assert json.loads((b/'acceptance_summary.json').read_text())['status']=='technical_acceptance_pass'
selected=[p for p in b.iterdir() if p.is_file() and p.name not in ('delivery_record.json','versioned_evidence_manifest.json')]
selected+=list((b/'validation_helpers').glob('*.py'))
selected+=list((b/'serialized_test_only').glob('*.json'))
assert all(p.suffix in ('.json','.md','.xml','.log','.py') for p in selected)
rows=[]
for p in sorted(selected):
 data=p.read_bytes();rows.append(dict(path=p.relative_to(b).as_posix(),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=dict(version='1.0.0',implementation_commit=commit,self_excluded=True,scope='Git evidence subset; full local synthetic fixture is intentionally not exported.',files=rows)
(b/'versioned_evidence_manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
selected.append(b/'versioned_evidence_manifest.json')
for p in selected:
 dst=target/p.relative_to(b);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
for name in ('startup_identity.json','implementation_supersession.json'):
 dst=dev/rel/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(main/rel/name,dst)
for row in rows:
 data=(target/row['path']).read_bytes();assert len(data)==row['size_bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
Path('/private/tmp/g14r20_evidence_paths.txt').write_text('\n'.join(str(p.relative_to(dev)) for p in sorted((dev/rel).rglob('*')) if p.is_file())+'\n')
print('exported',len(selected)+2,'files',sum(r['size_bytes'] for r in rows),'bytes')
