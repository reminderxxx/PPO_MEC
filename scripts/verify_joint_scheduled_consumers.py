"""Read-only registry→checkpoint→raw-event replay on completed engineering smokes."""
from pathlib import Path
import hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.agents.registry import build_agent
from scripts.train_joint_parameterized_control import episode
from scripts.launch_eight_vehicle_joint_experiment import arguments,_load_protocol,_hash_file
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts

def main():
    torch.set_num_threads(1)
    roots=sys.argv[1:] or ['joint_scheduled_final_smoke/main','joint_scheduled_final_smoke/baselines','joint_scheduled_clean_smoke']
    source=json.loads((ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json').read_text())
    splits,workflows,rsus=_load_protocol(arguments(source['command'][1:]));window=splits['dev'][0];report=[]
    for root in roots:
        paths=list((ROOT/'artifacts'/root).glob('*/manifest.json'));assert len(paths)==1
        path=paths[0];m=json.loads(path.read_text());assert m['status']=='complete' and m['scheduled_controls']
        original={}
        for line in (path.parent/'events.jsonl').open():
            e=json.loads(line)
            if e['frame_offset']==window['frame_offset']:
                original.setdefault((e['agent'],e['seed'],e['episode']),[]).append({k:v for k,v in e.items() if k not in ('agent','seed','episode','frame_offset')})
        for c in m['checkpoints']:
            if c['episode']!=m['training']['episodes']:continue
            checkpoint=path.parent/c['path'];assert _hash_file(checkpoint)==c['sha256'];saved=torch.load(checkpoint,weights_only=False)
            fresh=build_agent(saved['options']['method'],control_contract=m['action_contract'],scheduled_controls=True,**{k:v for k,v in saved['options'].items() if k!='method'})
            initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in fresh.state_dict().values())).hexdigest();assert initial==c['initial_tensor_sha256']
            policy=type(fresh).from_checkpoint(checkpoint)
            provider=LearnedMobilityPredictor(torch.load(path.parent/'predictor.pt',weights_only=False))
            env=MultiRequestTargetControlEnv(frames=window['frames'],vehicle_workflows=dict(zip(window['vehicle_ids'],workflows)),rsu_states=rsus,costs=ReplayCosts(**m['costs']['replay_costs']),prediction_provider=provider,**m['environment_options'])
            row,_=episode(env,policy,deterministic=True)
            assert json.loads(json.dumps(env.events))==original[(c['agent'],c['seed'],c['episode'])]
            report.append(dict(artifact=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=c['sha256'],producer_dirty=m['git_dirty'],agent=c['agent'],initial_tensor_identity=True,events_equal=True,reward=row['reward']))
    target=ROOT/'artifacts/joint_coordinated_audit_20261015/scheduled_consumer_smoke_check.json'
    target.write_text(json.dumps(dict(scope='engineering smoke consumer validation only, no performance claim',reproductions=report),indent=2)+'\n')
    print(json.dumps(dict(report=str(target),reproductions=len(report),all_events_equal=True)))
if __name__=='__main__':main()
