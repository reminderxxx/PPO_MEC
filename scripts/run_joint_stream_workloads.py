"""Train-only, frozen factorial workload/caching engineering experiment; zero RL."""
import argparse,json,sys,subprocess,itertools
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import _hash_file, arguments, NGSIMProvider
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.joint_training_scenarios import mobile_cohort
from src.envs.core.joint_stream_workloads import SIZE_BINS, ARRIVAL_MODES,request_plan,catalog_from_pool,capacity_lower_bounds
from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.core.joint_stream_rules import StreamCacheRule
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source_protocol',type=Path,default=ROOT/'artifacts/joint_delivery_control_v2/protocol/manifest.json');p.add_argument('--output_root',type=Path,default=ROOT/'artifacts/joint_stream_workloads_v3');p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty and not a.smoke:raise ValueError('full engineering run requires clean source')
    protocol=json.loads(a.source_protocol.read_text());source_path=Path(protocol['source_manifest']);source=json.loads(source_path.read_text())
    assert _hash_file(source_path)==protocol['source_manifest_sha256']
    original=arguments(source['command'][1:])
    for key,path in [('ngsim',original.mobility_csv_path),('alibaba',original.workflow_csv_path)]:assert _hash_file(path)==source['raw_data_sha256'][key]
    assert _hash_file(a.source_protocol.parent/'protocol.pt')==protocol['protocol_sha256'] and _hash_file(a.source_protocol.parent/'predictor.pt')==protocol['predictor_sha256']
    data=torch.load(a.source_protocol.parent/'protocol.pt',weights_only=False);bundle=torch.load(a.source_protocol.parent/'predictor.pt',weights_only=False)
    windows=[w for w in data['scenarios']['train'] if w['panel']=='base'];train_times={(f['source_segment_id'],f['time_index']) for w in windows for f in w['frames']}
    raw=NGSIMProvider(original.mobility_csv_path,max_rows=original.max_mobility_rows).get_loaded_frames()
    timeline={(f['source_segment_id'],f['time_index']):f for f in raw if (f['source_segment_id'],f['time_index']) in train_times};del raw
    sessions=[]
    for j in [0,4]:
        selected=windows[j:j+4];frames=[timeline[(f['source_segment_id'],f['time_index'])] for w in selected for f in w['frames']]
        assert len(frames)==480 and all(frames[i+1]['time_index']-frames[i]['time_index']==100 for i in range(479))
        fs,ids=mobile_cohort(frames,selected[0]['rsus']);sessions.append(dict(session_id=f'train_{selected[0]["frame_offset"]}_480',frames=fs,vehicle_ids=ids,rsus=selected[0]['rsus']))
    b=WorkflowDatasetBuilder();samples=b.build_alibaba_samples(original.workflow_csv_path,limit_jobs=256,min_tasks=5,max_tasks=40,adapter_assignment_profile='semantic_ai_service');pool=[]
    for lo,hi in SIZE_BINS.values():
        chosen=[s for s in samples if lo<=len(s['nodes'])<=hi][:4]
        if not chosen:raise ValueError('declared Alibaba stratum unavailable')
        pool.extend(b.sample_to_workflow_state(s) for s in chosen)
    grid=list(itertools.product(range(2),SIZE_BINS,ARRIVAL_MODES,[1,2],[4096,8192]))
    if a.smoke:grid=[(0,'short',mode,2,4096) for mode in ARRIVAL_MODES]
    home=a.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_engineering');home.mkdir(parents=True,exist_ok=False)
    rules=['reactive','naive_prefetch','gain_prefetch'];cases=[]
    options=dict(protocol['environment_options']);options.pop('model_cache_config',None)
    for i,(index,size,mode,families,capacity) in enumerate(grid):
        session=sessions[index];arrivals=request_plan(pool,session['vehicle_ids'],size,mode,families);catalog=catalog_from_pool(pool,families)
        cfg=dict(options,model_cache_config={**protocol['environment_options']['model_cache_config'],'capacity_bytes':capacity})
        cases.append(dict(case_id=f'c{i:03d}',session=index,size=size,arrival_mode=mode,families=families,capacity=capacity,arrivals=arrivals,catalog=catalog,options=cfg,
            bounds=capacity_lower_bounds(arrivals,480,4,cfg)))
    metadata=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_dirty=dirty,smoke=a.smoke,command=sys.argv,
        protocol='joint_stream_workload_engineering_v3',source_protocol=str(a.source_protocol.resolve()),source_protocol_sha256=_hash_file(a.source_protocol),raw_data_sha256=source['raw_data_sha256'],predictor_sha256=protocol['predictor_sha256'],
        data_selection='first two nonoverlapping four-window blocks from original train intervals; eight continuous covered vehicles ranked by geometry only; first4 Alibaba jobs/bin in fixed256job prefix',
        source_workflows=[w.to_dict() for w in pool],planned_cells=len(cases)*len(rules),rules=rules,training_episodes=0,paper_claim_ready=False,formal_holdout='sealed_not_evaluated',
        controls='common sticky reachable-balanced placement, causal GRU/shared state and mandatory resource protection; only model-prefetch rule differs; no state-preparation experiment',
        limits=['synthetic model/adapter byte mapping, not measured','serial DAG execution and FIFO per vehicle','long stratum may contain one source DAG','not independent paper windows, no learned baseline ranking','survival weight is a heuristic, not calibrated probability; known resources only'])
    metadata['cases']=[{k:([dict(frame=x.frame,vehicle_id=x.vehicle_id,request_id=x.request_id,workflow=x.workflow.to_dict()) for x in v] if k=='arrivals' else [asdict(o) for o in v] if k=='catalog' else v) for k,v in c.items() if k!='options'} for c in cases]
    metadata['sessions']=[dict(session_id=s['session_id'],time_start=s['frames'][0]['time_index'],time_end=s['frames'][-1]['time_index'],source_segment_id=s['frames'][0]['source_segment_id'],vehicle_ids=s['vehicle_ids']) for s in sessions]
    metadata['source_fingerprints']={str(p.relative_to(ROOT)):_hash_file(p) for p in list((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/audit_joint_stream_workloads.py']}
    manifest=home/'manifest.json';manifest.write_text(json.dumps(metadata,indent=2)+'\n');torch.save(dict(sessions=sessions,cases=cases),home/'protocol.pt');metadata['protocol_sha256']=_hash_file(home/'protocol.pt');manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    (home/'command.json').write_text(json.dumps(sys.argv,indent=2)+'\n');done=0
    with (home/'results.jsonl').open('w') as rows,(home/'events.jsonl').open('w') as events,(home/'receipts.jsonl').open('w') as receipts:
        for c in cases:
            session=sessions[c['session']]
            for rule_name in rules:
                env=MultiRequestStreamEnv(frames=session['frames'],arrivals=c['arrivals'],catalog=c['catalog'],rsu_states=session['rsus'],costs=ReplayCosts(**protocol['costs']['replay_costs']),prediction_provider=LearnedMobilityPredictor(bundle),**c['options'])
                rule=StreamCacheRule(rule_name);reward=0.;identity=dict(case_id=c['case_id'],rule=rule_name)
                while not env._control_episode_done():
                    decisions=env.decision_states();controls=rule.act(decisions,env.frame_index);_,value,_,info=env.step_controls(controls);reward+=value
                    for e in info['events']:events.write(json.dumps({**identity,**e,'control_planning':rule.last_diagnostics[e['vehicle_id']]})+'\n')
                    receipts.write(json.dumps({**identity,**info['frame_cost_receipt']})+'\n')
                rows.write(json.dumps({**identity,'reward':reward,'size':c['size'],'arrival_mode':c['arrival_mode'],'families':c['families'],'capacity':c['capacity'],'bounds':c['bounds'],'configuration':env.configuration(),**env.summary()})+'\n');rows.flush();events.flush();receipts.flush();done+=1
                print(json.dumps(dict(completed=done,planned=metadata['planned_cells'],case_id=c['case_id'],rule=rule_name)),flush=True)
    metadata.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),output_sha256={n:_hash_file(home/n) for n in ['results.jsonl','events.jsonl','receipts.jsonl']});manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(output=str(home))),flush=True)


if __name__=='__main__':main()
