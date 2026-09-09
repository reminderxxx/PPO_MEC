import sys,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/private/tmp/ppo_mec_g14r20_b_62f432c/scripts')
from continuation_executor.identity import strict_json,canonical,file_hash
b=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
counts=dict(json=0,jsonl=0,jsonl_records=0,xml=0);excluded={'artifact_integrity.json','format_validation.json','delivery_record.json','versioned_evidence_manifest.json'}
paths=[p for p in sorted(b.rglob('*')) if p.is_file() and '.git' not in p.relative_to(b).parts and p.name not in excluded]
for p in paths:
 if p.suffix=='.json':strict_json(p.read_text(encoding='utf-8-sig'));counts['json']+=1
 elif p.suffix=='.jsonl':
  for line in p.read_text().splitlines():strict_json(line);counts['jsonl_records']+=1
  counts['jsonl']+=1
 elif p.suffix=='.xml':ET.parse(p);counts['xml']+=1
report=dict(status='pass',checked_at=datetime.now(timezone.utc).isoformat(),counts=counts,duplicate_keys_forbidden=True,nonfinite_json_forbidden=True,scope='all JSON/JSONL/XML in the acceptance directory; Git internals excluded; format report additionally round-tripped')
raw=canonical(report)+b'\n';strict_json(raw);(b/'format_validation.json').write_bytes(raw)
paths.append(b/'format_validation.json')
rows=[dict(path=p.relative_to(b).as_posix(),size_bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(paths)]
manifest=dict(version='1.0.0',implementation_commit=b.name,status='pass',self_excluded=True,excluded=['.git/**','artifact_integrity.json','delivery_record.json','versioned_evidence_manifest.json'],files=rows)
raw=canonical(manifest)+b'\n';strict_json(raw);(b/'artifact_integrity.json').write_bytes(raw)
for row in rows:
 p=b/row['path'];assert p.stat().st_size==row['size_bytes'] and file_hash(p)==row['sha256']
print('strict format/hash pass',counts,'files',len(rows))
