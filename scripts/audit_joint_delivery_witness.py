"""Replay the frozen real negative case, then inspect protected reactive demand."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_joint_tail_control import make_env
from scripts.launch_eight_vehicle_joint_experiment import _hash_file
from src.envs.core.multi_request_target_control_env import JointRequestControl
from src.envs.core.joint_delivery_planning import reactive_link_calendar,delivery_migration_opportunities


def main():
    torch.set_num_threads(1);root=ROOT/'artifacts/joint_tail_control';case=json.loads((root/'prediction_negative_case.json').read_text())
    protocol=json.loads((root/'protocol/manifest.json').read_text());data=torch.load(root/'protocol/protocol.pt',weights_only=False)
    data['predictor']=torch.load(root/'protocol/predictor.pt',weights_only=False)
    assert _hash_file(root/'protocol/protocol.pt')==protocol['protocol_sha256']
    window=next(w for w in data['scenarios']['dev'] if w['scenario_id']==case['scenario']);reports=[];proof=[]
    for mode in ('on','off'):
        env=make_env(window,data,protocol);total=0.;rows=case['all_case_events'][mode]
        for frame in range(len(window['frames'])):
            events=[e for e in rows if e['frame_index']==frame];states=env.decision_states();controls={}
            for e in events:
                c=dict(e['control']);c['evict_object_ids']=tuple(c['evict_object_ids']);c['link_reservations']=tuple(map(tuple,c['link_reservations']))
                controls[e['vehicle_id']]=JointRequestControl(**c)
            if mode=='on' and frame in (11,12,15):
                targets={v:c.execution_target for v,c in controls.items()};cal=reactive_link_calendar(states,targets,8)
                ego='peachtree:244';plans=delivery_migration_opportunities(states[ego]['semantic_state'],cal,decisions=states,targets=targets)
                assert controls[ego].prepare_target=='rsu_d' and cal['rsu_d'][0] is not None and 'rsu_d' not in plans
                proof.append(dict(frame=frame,old_prepare=controls[ego].prepare_target,protected_current_slot=cal['rsu_d'][0],new_prepare_allowed=False,calendar=cal))
            _,reward,done,info=env.step_controls(controls);total+=reward
            reference={e['vehicle_id']:e for e in events}
            for e in info['events']:
                assert e['status']==reference[e['vehicle_id']]['status']
                assert e['work_progress']==reference[e['vehicle_id']]['work_progress']
        expected=35.473390625 if mode=='on' else 37.404415039062464
        assert abs(total-expected)<1e-8
        reports.append(dict(mode=mode,reward=total,completed_nodes=env.summary()['completed_nodes'],exact_status_progress=True))
    out=ROOT/'artifacts/joint_delivery_witness';out.mkdir(exist_ok=False)
    report=dict(source_commit='532dca6',source_case_sha256=_hash_file(root/'prediction_negative_case.json'),proof=proof,replays=reports,
        training_added=0,scope='same old prescribed actions reproduce old loss; new calendar forbids the three offending preparations. No claim of new-policy reward gain.')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
