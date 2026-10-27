"""One prospective main-only complete-session MC confirmation.

The matched budget is external sessions and their maximum horizon. Actual
terminal lengths remain visible; no truncated tail or post-terminal padding.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash,select_train_cases,run_episode
from scripts.run_stream_intent_study import VARIANTS
from scripts.train_joint_stream_control import make_env,actor_hash
from scripts.train_joint_tail_control import tensor_hash
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent as Main

CONTRACT='stream_complete_session_confirmation_v1'
_OWNED_OUTPUT=None


def complete_train_unit(case,data,manifest,agent,maximum_frames,trace,targets,identity):
    """One actual terminal trajectory and its auditable full native MC labels."""
    env=make_env(case,data,manifest);agent.reset_controller()
    if not 0<len(env.frames)<=maximum_frames<=480:
        raise ValueError('complete physical session must fit the declared frame cap')
    if env.frame_index!=0:raise ValueError('one fresh physical session required')
    rewards=[];rolls=[];decision_rows=0
    while not env._control_episode_done():
        frame=env.frame_index
        if frame>=maximum_frames:raise ValueError('nonterminal trajectory exceeded hard cap')
        released=sorted(env._released)
        controls,roll=agent.act_controls(env.decision_states(),frame)
        _,reward,_,info=env.step_controls(controls)
        assert env.frame_index==frame+1
        assert abs(reward-sum(info['frame_cost_receipt']['reward_components'].values()))<1e-8
        rewards.append(reward)
        if roll is not None:
            by={e['vehicle_id']:e for e in info['events']}
            roll.update(frame_index=frame,episode_end=False,initial_request_count=len(env.workflows))
            roll['cost_targets']=torch.tensor([[sum(np.ceil(
                env.model_cache.objects[k].transfer_size_bytes/env.model_cache.transfer_bytes_per_frame)
                for k in by[v]['cache_request'].get('admitted_object_ids',[]))/10,
                by[v]['state_transfer_link_frames']/10] for v in roll['vehicle_ids']],dtype=torch.float32)
            rolls.append(roll);decision_rows+=len(roll['actions'])
        trace.write(json.dumps(dict(**identity,unit_frame=frame,frame_index=frame,
            released_request_ids_before=released,reward=reward,
            receipt=info['frame_cost_receipt'],events=info['events']))+'\n')
    frames=len(rewards);last_release=max(a.frame for a in env.arrivals)
    assert 0<frames<=maximum_frames and frames>last_release
    assert env._released=={a.request_id for a in env.arrivals}
    suffix=0.;tails={}
    for frame,value in reversed(list(enumerate(rewards))):
        suffix=value/10+suffix;tails[frame]=suffix
    for roll in rolls:
        roll['returns']=torch.full_like(roll['values'],tails[roll['frame_index']])
        roll['team_credit_targets']=roll['returns']/len(env.workflows)
        roll['credit_targets']=roll['team_credit_targets'].clone()
    if rolls:rolls[-1]['episode_end']=True
    for roll in rolls:
        record=dict(**identity,frame_index=roll['frame_index'],vehicle_ids=roll['vehicle_ids'],
            request_ids=roll['request_ids'],episode_end=roll['episode_end'])
        for key in ['actions','head_logs','active_heads','values','returns','credit_targets',
                    'team_credit_targets','cost_targets','prefix_values','counterfactual_baselines']:
            record[key]=roll[key].detach().cpu().tolist()
        targets.write(json.dumps(record)+'\n')
    return dict(actual_frames=frames,decision_rows=decision_rows,reward=sum(rewards),
        physical_terminal=True,bootstrap_team_tail=0.,last_scheduled_release_frame=last_release,
        released_request_ids=sorted(env._released),summary=env.summary()),rolls


def load_inputs(parent_runs,diagnosis):
    parents={};source_path=None
    for run in parent_runs:
        plan=json.loads((run/'protocol.json').read_text())
        assert (run/'complete.json').exists()
        study=json.loads((Path(plan['study'])/'protocol.json').read_text())
        path=Path(study['parent_protocol'])
        assert source_path is None or source_path==path
        source_path=path;cp=plan['checkpoints']['relocation']
        assert file_hash(cp['path'])==cp['sha256']
        parents[plan['seed']]=dict(**cp,source_run=str(run.resolve()),
            source_run_protocol_sha256=file_hash(run/'protocol.json'))
    assert set(parents)=={7,17,27}
    source=json.loads(source_path.read_text())
    assert file_hash(source_path.parent/'protocol.pt')==source['assets']['protocol.pt']
    for name,h in source['source_fingerprints'].items():
        if name.startswith('src/') and name!='src/agents/sa_ghmappo_agent.py':assert file_hash(ROOT/name)==h,name
    report=json.loads(diagnosis.read_text())
    assert report['units']==144 and report['stream_units']==72
    assert report['additional_finetuning_declared_requests_after_cut']==576
    return parents,source_path,source,torch.load(source_path.parent/'protocol.pt',weights_only=False)


def main():
    global _OWNED_OUTPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent_runs',type=Path,nargs=3,required=True)
    p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[7,17,27],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty and not a.preflight:raise ValueError('clean frozen source required')
    parents,source_path,source,data=load_inputs(a.parent_runs,a.diagnosis)
    cases=select_train_cases(data);initial_ids=['train_000','train_038','train_067']
    variants=VARIANTS;epochs=4
    if a.preflight:
        # Input-selected stream case exercises the late wave; scores never tune this choice.
        if a.seed!=7:raise ValueError('single seed7 software preflight only')
        cases=[cases[1]];initial_ids=[cases[0]['case_id']]
        variants={v:VARIANTS[v] for v in ['control','intent']};epochs=1
    else:
        if a.campaign is None:raise ValueError('prospective single-launch campaign required')
        campaign=json.loads(a.campaign.read_text())
        assert campaign['contract']==CONTRACT and campaign['source_commit']==git
        assert campaign['seeds']==[7,17,27] and campaign['variants']==list(VARIANTS)
        assert campaign['training_case_ids']==[c['case_id'] for c in cases]
        assert campaign['maximum_training_frames']==69120 and campaign['maximum_scientific_launches']==1
        assert campaign['seed_outputs'][str(a.seed)]==str(a.output.resolve())
        with (a.campaign.parent/f'seed{a.seed}_execution_claim.json').open('x') as claim:
            json.dump(dict(pid=os.getpid(),claimed_at=datetime.now(timezone.utc).isoformat(),
                command=sys.argv,campaign_sha256=file_hash(a.campaign)),claim,indent=2)
    assert all(0<len(data['sessions'][c['session']]['frames'])<=480 for c in cases)
    a.output.mkdir(parents=True,exist_ok=False)
    _OWNED_OUTPUT=a.output
    (a.output/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),created_at=datetime.now(timezone.utc).isoformat(),command=sys.argv),indent=2)+'\n')
    plan=dict(contract=CONTRACT,created_at=datetime.now(timezone.utc).isoformat(),source_commit=git,
        git_dirty=dirty,command=sys.argv,seed=a.seed,variants=variants,preflight=a.preflight,
        parent_protocol=str(source_path),parent_protocol_sha256=file_hash(source_path),parents=parents,
        diagnosis=str(a.diagnosis.resolve()),diagnosis_sha256=file_hash(a.diagnosis),
        campaign=str(a.campaign.resolve()) if a.campaign else None,
        campaign_sha256=file_hash(a.campaign) if a.campaign else None,
        training_cases=[c['case_id'] for c in cases],training_episodes_per_cell=len(cases),
        maximum_frames_per_episode=480,maximum_training_frames_per_cell=480*len(cases),
        update_calls_per_cell=(len(cases)+3)//4,update_group_size=4,epochs=epochs,parameter_count=73940,
        initial_evaluation_cases=initial_ids,final_evaluation_cases=[c['case_id'] for c in cases],
        main_source_sha256=file_hash(ROOT/'src/agents/sa_ghmappo_agent.py'),
        runner_source_sha256=file_hash(Path(__file__)),
        environment_source_sha256=file_hash(ROOT/'src/envs/core/multi_request_stream_env.py'),
        baseline_source_sha256=file_hash(ROOT/'src/envs/core/joint_stream_rules.py'),
        return_contract='complete observed native team suffix/10, accounting credit/8, no cut bootstrap',
        budget_contract='same external sessions and maximum horizon, actual terminal lengths may differ',
        evaluation='greedy complete physical session; all initials and fixed finals retained, no selection',
        warm_start='same frozen relocation parent tensors per seed, fresh optimizers and reset schedule',
        internal_gate=dict(mean_reward_delta_min=0,mean_completed_delta_min=0,
            unused_optional_bytes_reduction_min=.20,per_seed_mean_completed_delta_min=-.25),
        limits=['Consumed training times only, no formal/holdout',
            'Matched external sessions and maximum budget, not identical realized frames or gradient steps',
            'Inherited full-model parents, not from-scratch module necessity',
            'Prediction deletion also changes planner masks; optional intent restricts policy choices'],paper_claim_ready=False)
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    by_id={c['case_id']:c for c in cases}
    with (a.output/'cells.jsonl').open('x') as cells:
        for variant,options in variants.items():
            home=a.output/f'{variant}_seed{a.seed}';home.mkdir()
            random.seed(a.seed);np.random.seed(a.seed)
            agent=Main.from_relocation_checkpoint(parents[a.seed]['path'],training_budget=len(cases),**options)
            assert sum(p.numel() for p in agent.parameters())==73940
            assert not agent.optimizer.state and not agent.surrogate_optimizer.state
            meta=dict(variant=variant,seed=a.seed,status='running',actual_frames=0,training_episodes=0,
                parameter_count=73940,updates=[],checkpoints=[],initial_tensor_sha256=tensor_hash(agent),
                initial_actor_sha256=actor_hash(agent))
            agent.save_checkpoint(home/'initial.pt')
            meta['checkpoints'].append(dict(phase='initial',path='initial.pt',sha256=file_hash(home/'initial.pt')))
            with (home/'evaluation.jsonl').open('x') as ev:
                for cid in initial_ids:
                    row=run_episode(make_env(by_id[cid],data,source),agent,home/f'initial_{cid}.jsonl',learned=True)
                    ev.write(json.dumps(dict(phase='initial',case_id=cid,**row))+'\n');ev.flush()
                torch.manual_seed(a.seed+20000);buffer=[]
                with (home/'training.jsonl').open('x') as log,(home/'training_trace.jsonl').open('x') as trace,(home/'training_targets.jsonl').open('x') as targets:
                    for i,c in enumerate(cases):
                        started=time.monotonic()
                        row,rolls=complete_train_unit(c,data,source,agent,480,trace,targets,
                            dict(seed=a.seed,variant=variant,case_id=c['case_id']))
                        buffer.extend(rolls);meta['actual_frames']+=row['actual_frames'];meta['training_episodes']+=1
                        update=None
                        if (i+1)%4==0 or i==len(cases)-1:
                            update=agent.update(buffer,epochs=epochs);buffer=[];meta['updates'].append(update)
                        log.write(json.dumps(dict(case_id=c['case_id'],**row,update=update))+'\n')
                        for stream in [log,trace,targets]:stream.flush()
                        (home/'progress.json').write_text(json.dumps(dict(case_index=i+1,**meta),indent=2)+'\n')
                        print(json.dumps(dict(seed=a.seed,variant=variant,phase='train',case=c['case_id'],
                            frames=meta['actual_frames'],episodes=meta['training_episodes'],seconds=time.monotonic()-started)),flush=True)
                assert meta['training_episodes']==len(cases) and meta['actual_frames']<=plan['maximum_training_frames_per_cell']
                assert len(meta['updates'])==plan['update_calls_per_cell']
                agent.save_checkpoint(home/'final.pt');meta['checkpoints'].append(dict(phase='final',path='final.pt',sha256=file_hash(home/'final.pt')))
                evaluated=Main.from_checkpoint(home/'final.pt');assert tensor_hash(agent)==tensor_hash(evaluated)
                for c in cases:
                    row=run_episode(make_env(c,data,source),evaluated,home/f'final_{c["case_id"]}.jsonl',learned=True)
                    ev.write(json.dumps(dict(phase='final',case_id=c['case_id'],**row))+'\n');ev.flush()
            meta.update(status='complete',final_tensor_sha256=tensor_hash(agent),final_actor_sha256=actor_hash(agent),
                output_sha256={f:file_hash(home/f) for f in ['training.jsonl','training_trace.jsonl','training_targets.jsonl','evaluation.jsonl']})
            (home/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
            cells.write(json.dumps(dict(path=home.name,**meta))+'\n');cells.flush()
            print(json.dumps(dict(seed=a.seed,variant=variant,phase='cell_complete')),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',cells=len(variants),cells_sha256=file_hash(a.output/'cells.jsonl')),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        if _OWNED_OUTPUT is not None:
            with (_OWNED_OUTPUT/'failure.json').open('x') as stream:
                json.dump(dict(failed_at=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()),stream,indent=2)
        raise
