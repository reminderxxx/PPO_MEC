"""Predeclare the existing fixed mobility stratum and evaluate new delivery models."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import joint_tail_mobility_support as consumer
from scripts.train_joint_delivery_control import make_agent,NAMES
from scripts.launch_eight_vehicle_joint_experiment import _hash_file


def prepare(root):
    plan=json.loads((root/'comparison_plan.json').read_text());parent=ROOT/'artifacts/joint_tail_control/mobility_support/manifest.json'
    old=json.loads(parent.read_text());m=json.loads((root/'protocol/manifest.json').read_text());home=root/'mobility_support';home.mkdir(exist_ok=False)
    assert _hash_file(parent.parent/'protocol.pt')==old['protocol_sha256'] and old['predictor_sha256']==m['predictor_sha256']
    shutil.copyfile(parent.parent/'protocol.pt',home/'protocol.pt')
    metadata=dict(m,scenarios=old['scenarios'],protocol_sha256=old['protocol_sha256'],predictor_source=str((root/'protocol/predictor.pt').resolve()),
        parent_mobility_manifest=str(parent),parent_mobility_manifest_sha256=_hash_file(parent),training_episodes_added=0,
        producer_sha256=_hash_file(Path(__file__)))
    (home/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    support=dict(defined_at=datetime.now(timezone.utc).isoformat(),kind='predeclared_delivery_mobility_support_v1',agents=plan['primary_agents']+plan['support_agents'],
        actor_source_commit=plan['source_commit'],episode=128,seeds=plan['seeds'],training_episodes_added=0,
        primary_plan_sha256=_hash_file(root/'comparison_plan.json'),
        condition='byte-identical previous fixed original-radius two-site-covered high-mobility cohort; no new score selection',
        limits='predeclared secondary dev stratum, previously exploratory; same original eight intervals, not independent new windows; no formal/holdout')
    (root/'mobility_support_plan.json').write_text(json.dumps(support,indent=2)+'\n');print(json.dumps(support),flush=True)


def collect(root):
    plan=json.loads((root/'mobility_support_plan.json').read_text());checked=0
    from collections import defaultdict
    from scripts.audit_joint_delivery_resources import audit_optional_cache_evictions
    data,_=consumer.load_protocol(root);windows={w['scenario_id']:w for w in data['scenarios']['dev']}
    for name in plan['agents']:
        grouped=defaultdict(list);events=[]
        for line in (root/'mobility_support'/name/'events.jsonl').open():
            e=json.loads(line);events.append(e);grouped[(e['seed'],e['scenario_id'],e['mode'],e['frame_index'])].append(e)
        for e in events:
            workflows=dict(zip(windows[e['scenario_id']]['vehicle_ids'],data['workflows']))
            audit_optional_cache_evictions(e,grouped[(e['seed'],e['scenario_id'],e['mode'],e['frame_index'])],workflows)
        for line in (root/'mobility_support'/name/'events.jsonl').open():
            e=json.loads(line);c=e['control'];d=e['control_planning'];vid=e['vehicle_id']
            if c['prepare_target'] is not None:assert d['link_calendar'][c['prepare_target']][0] is None
            if c['cache_target'] is not None:
                span=d['cache_admission_span'];cal=d['link_calendar'][c['cache_target']]
                assert span<=len(cal) and all(x is None or x==[vid,'model'] for x in cal[:span])
            checked+=1
    consumer.collect(root);path=root/'mobility_support/audit.json';report=json.loads(path.read_text());report['delivery_protection_events_checked']=checked
    path.write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','evaluate','collect']);p.add_argument('--root',type=Path,required=True);p.add_argument('--agent',choices=NAMES);a=p.parse_args()
    consumer.torch.set_num_threads(1);consumer.make_agent=make_agent;root=a.root.resolve()
    if a.mode=='prepare':prepare(root)
    elif a.mode=='evaluate':consumer.evaluate(root,a.agent)
    else:collect(root)


if __name__=='__main__':main()
