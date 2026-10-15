"""Predeclared, zero-training mobility stratum; never replaces the frozen main table."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import json
from math import hypot
from pathlib import Path
import statistics
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import arguments,_hash_file
from scripts.train_joint_parameterized_control import episode
from scripts.train_joint_tail_control import make_agent,make_env,tensor_hash,NAMES
from src.data.mobility.ngsim_provider import NGSIMProvider
from src.envs.core.joint_training_scenarios import mobile_cohort,coverage_counts,public_scenarios


def select_cohort(frames,sites,count=8):
    eligible=None
    for frame in frames:
        covered={v.vehicle_id for v in frame['vehicles'] if sum(
            hypot(v.position_x-r.position_x,v.position_y-r.position_y)<=r.coverage_radius for r in sites)>=2}
        eligible=covered if eligible is None else eligible&covered
    filtered=[dict(f,vehicles=[v for v in f['vehicles'] if v.vehicle_id in eligible]) for f in frames]
    return mobile_cohort(filtered,sites,count=count)


def prepare(root):
    home=root/'mobility_support';home.mkdir(exist_ok=False)
    main=json.loads((root/'comparison_plan.json').read_text());parent=root/'protocol/manifest.json';m=json.loads(parent.read_text())
    assert _hash_file(parent)==main['prepared_manifest_sha256']
    assert _hash_file(parent.parent/'protocol.pt')==m['protocol_sha256']
    subprocess.run(['git','diff','--quiet',main['source_commit'],'--','src/',
                    'scripts/train_joint_tail_control.py','scripts/train_joint_parameterized_control.py'],cwd=ROOT,check=True)
    source=json.loads(Path(m['source_manifest']).read_text());args=arguments(source['command'][1:])
    # Definition is persisted before any policy is evaluated on these cohorts.
    plan=dict(defined_at=datetime.now(timezone.utc).isoformat(),kind='exploratory_mobility_support_v1',
        primary_plan_sha256=_hash_file(root/'comparison_plan.json'),parent_protocol_sha256=_hash_file(parent),
        actor_source_commit=main['source_commit'],agents=main['primary_agents']+main['support_agents'],seeds=main['seeds'],
        episode=128,training_episodes_added=0,condition='original radius1.2; all frames>=2 reachable sites; descending geometry departures/displacement/ID',
        selection='one fixed definition; failure is Unavailable; no policy-score search',
        primary_table='unchanged base/tight weights; supplementary exploratory stratum only',
        limits='same eight dev intervals, not independent windows; defined after main training began; no formal/holdout')
    (root/'mobility_support_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    assert _hash_file(args.mobility_csv_path)==m['raw_data_sha256']['ngsim']
    raw=NGSIMProvider(args.mobility_csv_path,max_rows=args.max_mobility_rows).get_loaded_frames()
    timeline={(f['source_segment_id'],f['time_index']):f for f in raw}
    data=torch.load(parent.parent/'protocol.pt',weights_only=False);windows=[]
    for w in data['scenarios']['dev']:
        if w['panel']!='base':continue
        frames=[timeline[(f['source_segment_id'],f['time_index'])] for f in w['frames']]
        selected,ids=select_cohort(frames,w['rsus'])
        windows.append(dict(w,scenario_id=f"dev_{w['frame_offset']}_mobile_base",panel='mobile_base',
                            frames=selected,vehicle_ids=ids,**coverage_counts(selected,w['rsus'])))
    bundle=dict(workflows=data['workflows'],scenarios=dict(train=[],dev=windows))
    torch.save(bundle,home/'protocol.pt')
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
    metadata=deepcopy(m);metadata.update(kind=plan['kind'],created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_dirty=bool(dirty),dirty_paths=dirty,
        actor_source_commit=main['source_commit'],predictor_source=str((parent.parent/'predictor.pt').resolve()),
        protocol_sha256=_hash_file(home/'protocol.pt'),scenarios=public_scenarios(bundle['scenarios']),
        producer_sha256=_hash_file(Path(__file__)),training_episodes_added=0)
    (home/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(home=str(home),geometric_departures=sum(w['coverage_departures'] for w in windows),scenes=len(windows)),indent=2))


def load_protocol(root):
    home=root/'mobility_support';m=json.loads((home/'manifest.json').read_text())
    assert _hash_file(home/'protocol.pt')==m['protocol_sha256']
    assert _hash_file(Path(m['predictor_source']))==m['predictor_sha256']
    data=torch.load(home/'protocol.pt',weights_only=False);data['predictor']=torch.load(m['predictor_source'],weights_only=False)
    assert public_scenarios(data['scenarios'])==m['scenarios']
    return data,m


def evaluate(root,name):
    plan=json.loads((root/'mobility_support_plan.json').read_text());data,protocol=load_protocol(root)
    subprocess.run(['git','diff','--quiet',plan['actor_source_commit'],'--','src/',
                    'scripts/train_joint_tail_control.py','scripts/train_joint_parameterized_control.py'],cwd=ROOT,check=True)
    path=next((root/name).glob('*/manifest.json'));m=json.loads(path.read_text())
    assert m['status']=='complete' and m['git_commit']==plan['actor_source_commit'] and not m['git_dirty']
    assert m['seeds']==plan['seeds'] and m['episodes']==plan['episode']
    output=root/'mobility_support'/name;output.mkdir(exist_ok=False);rows=[];bindings=[]
    with (output/'events.jsonl').open('w') as stream:
        for seed in plan['seeds']:
            c=next(c for c in m['checkpoints'] if c['seed']==seed and c['episode']==plan['episode'])
            cp=path.parent/c['path'];assert _hash_file(cp)==c['sha256']
            agent=type(make_agent(name,seed,m['episodes'])).from_checkpoint(cp);before=tensor_hash(agent)
            for mode in (['policy','prediction_off'] if name=='sa_ghmappo' else ['policy']):
                if mode=='prediction_off':agent.prediction_enabled=False
                for w in data['scenarios']['dev']:
                    identity=dict(agent=name,seed=seed,episode=plan['episode'],mode=mode,scenario_id=w['scenario_id'],frame_offset=w['frame_offset'],panel=w['panel'])
                    row,_=episode(make_env(w,data,protocol),agent,deterministic=True,event_file=stream,identity=identity)
                    rows.append(dict(identity,**row))
            assert tensor_hash(agent)==before and _hash_file(cp)==c['sha256']
            bindings.append(dict(seed=seed,checkpoint=str(cp.relative_to(ROOT)),sha256=c['sha256']))
    (output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'evaluation.json').write_text(json.dumps(dict(status='complete',agent=name,plan_sha256=_hash_file(root/'mobility_support_plan.json'),
        protocol_sha256=protocol['protocol_sha256'],checkpoints=bindings,training_episodes_added=0,
        events_sha256=_hash_file(output/'events.jsonl'),results_sha256=_hash_file(output/'results.jsonl')),indent=2)+'\n')
    print(json.dumps(dict(agent=name,cells=len(rows),complete=True)),flush=True)


def collect(root):
    plan=json.loads((root/'mobility_support_plan.json').read_text());data,protocol=load_protocol(root)
    summaries={};prediction=[];replay=[];total=0
    for name in plan['agents']:
        home=root/'mobility_support'/name;m=json.loads((home/'evaluation.json').read_text())
        assert m['status']=='complete' and m['plan_sha256']==_hash_file(root/'mobility_support_plan.json')
        assert _hash_file(home/'events.jsonl')==m['events_sha256'] and _hash_file(home/'results.jsonl')==m['results_sha256']
        rows=[json.loads(s) for s in (home/'results.jsonl').read_text().splitlines()];grouped=defaultdict(list)
        assert len(rows)==(48 if name=='sa_ghmappo' else 24)
        for line in (home/'events.jsonl').open():
            e=json.loads(line);grouped[(e['seed'],e['scenario_id'],e['mode'])].append(e);total+=1
        for r in rows:
            es=grouped[(r['seed'],r['scenario_id'],r['mode'])]
            receipts=[e['frame_cost_receipt'] for e in es if 'frame_cost_receipt' in e]
            assert abs(sum(sum(x['reward_components'].values()) for x in receipts)-r['reward'])<1e-8
            assert all(0<=v<=1 for x in receipts for v in x['link_usage_by_rsu'].values())
        policy=[r for r in rows if r['mode']=='policy'];assert len(policy)==24
        summaries[name]={k:statistics.mean(r[k] for r in policy) for k in ['reward','completed_requests','completed_nodes','model_transfer_link_frames','state_transfer_link_frames']}
        summaries[name]['reward_by_seed']=[statistics.mean(r['reward'] for r in policy if r['seed']==s) for s in plan['seeds']]
        if name=='sa_ghmappo':
            off={(r['seed'],r['scenario_id']):r for r in rows if r['mode']=='prediction_off'}
            prediction=[dict(seed=r['seed'],scenario_id=r['scenario_id'],delta_reward=r['reward']-off[(r['seed'],r['scenario_id'])]['reward'],delta_completed=r['completed_requests']-off[(r['seed'],r['scenario_id'])]['completed_requests']) for r in policy]
        for c in m['checkpoints']:
            cp=ROOT/c['checkpoint'];assert _hash_file(cp)==c['sha256']
            agent=type(make_agent(name,c['seed'],plan['episode'])).from_checkpoint(cp)
            env=make_env(data['scenarios']['dev'][0],data,protocol);row,_=episode(env,agent,deterministic=True)
            original=[{k:v for k,v in e.items() if k not in ('agent','seed','episode','mode','scenario_id','frame_offset','panel')} for e in grouped[(c['seed'],data['scenarios']['dev'][0]['scenario_id'],'policy')]]
            assert json.loads(json.dumps(env.events))==original
            replay.append(dict(agent=name,seed=c['seed'],events_equal=True))
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-15',target_venue='IEEE TMC',policy_version='tmc_review_policy_v3_20260621',
        actor_source_commit=plan['actor_source_commit'],evidence_level='exploratory dev raw/replayed; paper E0_UNAVAILABLE',summary=summaries,prediction_probe=prediction,
        counts=dict(policy_cells=240,prediction_off_cells=24,events=total,reproductions=len(replay)),reproductions=replay,
        limits=plan['limits'],primary_table_unchanged=True,training_episodes_added=0)
    (root/'mobility_support/audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['counts']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','evaluate','collect']);p.add_argument('--agent',choices=NAMES)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/joint_tail_control');a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='prepare':prepare(a.root.resolve())
    elif a.mode=='evaluate':evaluate(a.root.resolve(),a.agent)
    else:collect(a.root.resolve())


if __name__=='__main__':main()
