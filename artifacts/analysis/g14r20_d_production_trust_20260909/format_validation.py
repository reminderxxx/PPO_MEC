import pathlib,sys,json,datetime,xml.etree.ElementTree as ET
O=pathlib.Path('/private/tmp/g14r20_d_evidence');F=pathlib.Path('/private/tmp/synthetic_g14r20_d_exact_876c369')
sys.path.insert(0,str(O));from verify_delivery import load,strict,canonical
counts=dict(json=0,jsonl=0,jsonl_records=0,xml=0);paths=[]
for root in (F,O/'serialized_test_only'):
 for p in root.rglob('*'):
  if '.git' in p.parts or not p.is_file():continue
  if p.suffix=='.json':load(p);counts['json']+=1;paths.append(str(p))
  elif p.suffix=='.jsonl':
   for l in p.read_text().splitlines():strict(l);counts['jsonl_records']+=1
   counts['jsonl']+=1;paths.append(str(p))
for p in O.glob('*.json'):load(p);counts['json']+=1;paths.append(str(p))
for p in O.glob('*.xml'):ET.parse(p);counts['xml']+=1;paths.append(str(p))
d=dict(status='pass',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=counts,scope='Final compact evidence plus complete synthetic fixture; Git internals and superseded candidates excluded',duplicate_keys_forbidden=True,nonfinite_json_forbidden=True,paths=paths)
(O/'format_validation.json').write_bytes(canonical(d)+b'\n');load(O/'format_validation.json');print(json.dumps(d['counts']))
