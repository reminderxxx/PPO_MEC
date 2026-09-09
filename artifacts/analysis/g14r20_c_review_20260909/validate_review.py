from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,subprocess,datetime
O=Path(__file__).parent

def strict(s):
 def pairs(rows):
  d={}
  for k,v in rows:
   if k in d:raise ValueError('duplicate '+k)
   d[k]=v
  return d
 return json.loads(s,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
for p in O.glob('*.json'):strict(p.read_text())
for p in O.glob('*.xml'):ET.parse(p)
s=strict((O/'review_summary.json').read_text());assert s['verdict']['real_execution_authorized'] is False and s['verdict']['continuation_approval']=='not_issued'
x=strict((O/'next_action_packet.json').read_text());assert x['execution_authorized'] is False and x['status']=='unsigned_not_issued';assert x['exact_binding']['holdout_capability'] is False
assert x['exact_binding']['reviewed_executor_identity']['commit']==s['metadata']['executor_commit']
i=strict((O/'identity_integrity_recomputed.json').read_text());assert all(not i[k]['mismatches'] for k in ['versioned_evidence_manifest.json','artifact_integrity.json','protected_inventory']);assert not any(i['additions'].values());assert i['user_files_match_b'];assert all(v['matches'] for v in i['executor_files']+i['versioned_commit_comparison'])
probes=strict((O/'independent_probes.json').read_text());assert probes['production_signers']==0 and probes['b_monitor_recomputed']['all_zero'];assert all(probes[k]['exit_code']==0 for k in ['gate_false','gate_missing']);assert not probes['b_origins_recomputed']['mismatches']
assert all(x['argv_equal'] and x['cwd_equal'] and x['exit_code']==0 for x in probes['command_log_consistency'])
suite=next(ET.parse(O/'independent_pytest_retry.xml').getroot().iter('testsuite'));assert suite.get('tests')=='71' and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
files=[]
for p in sorted(O.iterdir()):
 if p.is_file() and p.name not in {'review_integrity_manifest.json','review_validation.json','delivery_record.json'}:
  files.append(dict(path=p.name,size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(O/'review_integrity_manifest.json').write_text(json.dumps({'files':files,'excluded':['review_integrity_manifest.json','review_validation.json','delivery_record.json'],'note':'No synthetic checkpoints or approval files included; exact review commit carries manifest identity'},indent=2)+'\n')
report={'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','json_count':len(list(O.glob('*.json'))),'xml_count':len(list(O.glob('*.xml'))),'manifest_files':len(files),'independent_boundary_tests':71,'protection_check':i['checked_at']}
(O/'review_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
