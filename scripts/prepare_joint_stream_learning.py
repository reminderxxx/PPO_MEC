"""Freeze score-blind original-train session splits and equal stream learning budgets."""
import argparse, itertools, json, subprocess, sys
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import _hash_file,arguments,NGSIMProvider
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.joint_training_scenarios import mobile_cohort
from src.envs.core.joint_stream_workloads import SIZE_BINS,request_plan,catalog_from_pool,capacity_lower_bounds
from scripts.train_joint_stream_control import NAMES
from src.envs.core.joint_learned_prediction import fit_predictor
from src.envs.core.joint_stream_refinement import balanced_training_schedule


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source_protocol',type=Path,default=ROOT/'artifacts/joint_delivery_control_v2/protocol/manifest.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty and not a.smoke:raise ValueError('clean frozen source required')
    parent=json.loads(a.source_protocol.read_text());source_path=Path(parent['source_manifest']);assert _hash_file(source_path)==parent['source_manifest_sha256']
    source=json.loads(source_path.read_text());original=arguments(source['command'][1:])
    for k,path in [('ngsim',original.mobility_csv_path),('alibaba',original.workflow_csv_path)]:assert _hash_file(path)==source['raw_data_sha256'][k]
    for name,key in [('protocol.pt','protocol_sha256'),('predictor.pt','predictor_sha256')]:assert _hash_file(a.source_protocol.parent/name)==parent[key]
    old=torch.load(a.source_protocol.parent/'protocol.pt',weights_only=False);windows=[w for w in old['scenarios']['train'] if w['panel']=='base']
    allowed={(f['source_segment_id'],f['time_index']) for w in windows for f in w['frames']}
    raw=NGSIMProvider(original.mobility_csv_path,max_rows=original.max_mobility_rows).get_loaded_frames()
    timeline={(f['source_segment_id'],f['time_index']):f for f in raw if (f['source_segment_id'],f['time_index']) in allowed};del raw
    sessions=[];seen=set()
    for start in [0,4,8,12,16]:
        block=windows[start:start+4];frames=[timeline[f['source_segment_id'],f['time_index']] for w in block for f in w['frames']]
        assert len(frames)==480 and all(frames[i+1]['time_index']-frames[i]['time_index']==100 for i in range(479))
        keys={(f['source_segment_id'],f['time_index']) for f in frames};assert keys<=allowed and not keys&seen;seen|=keys
        frames,ids=mobile_cohort(frames,block[0]['rsus'])
        sessions.append(dict(frame_offset=block[0]['frame_offset'],session_id=f'originaltrain_{block[0]["frame_offset"]}_480',frames=frames,vehicle_ids=ids,rsus=block[0]['rsus'],split='train' if start<12 else 'dev'))
    builder=WorkflowDatasetBuilder();samples=builder.build_alibaba_samples(original.workflow_csv_path,limit_jobs=256,min_tasks=5,max_tasks=40,adapter_assignment_profile='semantic_ai_service')
    pool=[]
    for lo,hi in SIZE_BINS.values():
        selected=[s for s in samples if lo<=len(s['nodes'])<=hi][:4]
        if not selected:raise ValueError('declared source stratum unavailable')
        pool.extend(builder.sample_to_workflow_state(s) for s in selected)
    grid=list(itertools.product(SIZE_BINS,['queued_batch','stream'],[1,2],[4096,8192]));cases={'train':[],'dev':[]}
    for i,session in enumerate(sessions):
        if a.smoke and i not in [0,3]:continue
        selected=[('short','queued_batch',2,4096)] if a.smoke else grid
        for size,mode,families,capacity in selected:
            index=len(cases[session['split']]);cid=f'{session["split"]}_{index:03d}'
            arrivals=request_plan(pool,session['vehicle_ids'],size,mode,families)
            options=dict(parent['environment_options']);options['model_cache_config']=dict(options['model_cache_config'],capacity_bytes=capacity)
            cases[session['split']].append(dict(case_id=cid,session=i,size=size,arrival_mode=mode,families=families,capacity=capacity,
                arrivals=arrivals,catalog=catalog_from_pool(pool,families),options=options,bounds=capacity_lower_bounds(arrivals,480,4,options)))
    # Short smoke is software validation, not a performance sample or budget selection.
    if a.smoke:
        for s in sessions:s['frames']=s['frames'][:32]
    seeds=[7] if a.smoke else [7,17,27];episodes=4 if a.smoke else 64
    schedule={}
    for j,seed in enumerate(seeds):
        schedule[str(seed)]=[cases['train'][0]['case_id']]*episodes if a.smoke else balanced_training_schedule(cases['train'], episodes)
    probes=[c['case_id'] for i,c in enumerate(cases['dev']) if i%4==0]
    # Refit solely inside the NEW train partition; old predictor saw newdev times.
    predictor_windows=[dict(s,source_segment_id=s['frames'][0]['source_segment_id'],time_start=s['frames'][0]['time_index'],time_end=s['frames'][-1]['time_index']) for i,s in enumerate(sessions) if s['split']=='train']
    bundle=fit_predictor(predictor_windows,seed=101,epochs=30,horizon=8)
    data=dict(sessions=sessions,cases=cases,predictor=bundle)
    a.output.mkdir(parents=True,exist_ok=False);torch.save(data,a.output/'protocol.pt');torch.save(bundle,a.output/'predictor.pt')
    manifest=dict(status='frozen',protocol='joint_stream_learning_v2',action_contract='joint_stream_control_v2',training_schedule_contract='balanced24interleaved_same_all_seeds_64sessions',baseline_context_contract='DAG_public_structure_DT_causal_mobility_distinct_v2',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_dirty=dirty,smoke=a.smoke,
        command=sys.argv,created_at=datetime.now(timezone.utc).isoformat(),source_protocol=str(a.source_protocol.resolve()),source_protocol_sha256=_hash_file(a.source_protocol),raw_data_sha256=source['raw_data_sha256'],parent_predictor_sha256=parent['predictor_sha256'],predictor_sha256=_hash_file(a.output/'predictor.pt'),predictor_fit_contract='newtrain_session0_1_fit_session2_internal_validation_seed101_epochs30_horizon8_no_newdev',predictor_fit_offsets=bundle['fit_offsets'],predictor_validation_offsets=bundle['validation_offsets'],
        assets={name:_hash_file(a.output/name) for name in ['protocol.pt','predictor.pt']},costs=parent['costs'],methods=NAMES,seeds=seeds,episodes=episodes,
        checkpoint_episodes=[4] if a.smoke else [32,64],curve_probe_ids=probes,training_schedule=schedule,
        training_session_count=len(seeds)*episodes*len(NAMES),maximum_training_frames=len(seeds)*episodes*len(NAMES)*(32 if a.smoke else 480),
        train_case_count=len(cases['train']),dev_case_count=len(cases['dev']),return_contract='undiscounted_actual_team_tail_scale0.1_per_fixed8vehicle_including_idle',
        primary_checkpoint='fixed_final_no_peak_selection',formal_holdout='sealed_not_read',paper_claim_ready=False,
        selection='originaltrain20 base windows grouped 5disjoint blocks; first3 train,last2dev; fixed first4realDAG/bin in256jobprefix, long only1',
        source_fingerprints={str(f.relative_to(ROOT)):_hash_file(f) for f in [*list((ROOT/'src').rglob('*.py')),*list((ROOT/'scripts').glob('*joint_stream*.py'))]})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(a.output/'command.json').write_text(json.dumps(sys.argv,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),train=len(cases['train']),dev=len(cases['dev']),episodes=manifest['training_session_count'],max_frames=manifest['maximum_training_frames'])))


if __name__=='__main__':main()
