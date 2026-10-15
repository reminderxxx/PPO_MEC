"""Reproduce the strong unmodified legacy cache controller on identical physics/data."""
import argparse,json,statistics,sys,torch
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_joint_tail_control import make_agent,make_env,tensor_hash
from scripts.train_joint_parameterized_control import episode
from scripts.launch_eight_vehicle_joint_experiment import _hash_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);root=a.root.resolve()
    plan=json.loads((root/'comparison_plan.json').read_text());ref=plan['legacy_cache_reference'];path=Path(ref['manifest']);assert _hash_file(path)==ref['manifest_sha256']
    original=json.loads(path.read_text());assert original['status']=='complete' and original['episodes']==128 and original['seeds']==plan['seeds']
    proto=json.loads((root/'protocol/manifest.json').read_text());oldproto=json.loads(Path(original['prepared_protocol']).read_text())
    assert proto['protocol_sha256']==oldproto['protocol_sha256'] and proto['predictor_sha256']==oldproto['predictor_sha256']
    assert proto['costs']==original['costs'] and proto['environment_options']==original['environment_options']
    data=torch.load(root/'protocol/protocol.pt',weights_only=False);data['predictor']=torch.load(root/'protocol/predictor.pt',weights_only=False)
    dev=[json.loads(s) for s in (path.parent/'dev.jsonl').read_text().splitlines()];expected={(r['seed'],r['scenario_id']):r for r in dev if r['episode']==128}
    events={}
    for line in (path.parent/'events.jsonl').open():
        e=json.loads(line);events.setdefault((e['seed'],e['scenario_id']),[]).append({k:v for k,v in e.items() if k not in ('agent','seed','episode','scenario_id','panel','frame_offset')})
    out=root/'legacy_cache_reference';out.mkdir(exist_ok=False);rows=[];bindings=[]
    with (out/'events.jsonl').open('w') as stream:
        for c in original['checkpoints']:
            if c['episode']!=128:continue
            cp=path.parent/c['path'];assert _hash_file(cp)==c['sha256'];policy=type(make_agent('cache_offload_drl',c['seed'],128)).from_checkpoint(cp);before=tensor_hash(policy)
            for w in data['scenarios']['dev']:
                env=make_env(w,data,proto);identity=dict(agent='cache_frozen_legacy',seed=c['seed'],episode=128,scenario_id=w['scenario_id'],frame_offset=w['frame_offset'],panel=w['panel'])
                row,_=episode(env,policy,deterministic=True,event_file=stream,identity=identity)
                assert abs(row['reward']-expected[(c['seed'],w['scenario_id'])]['reward'])<1e-8
                assert json.loads(json.dumps(env.events))==events[(c['seed'],w['scenario_id'])]
                rows.append(dict(identity,**row))
            assert before==tensor_hash(policy) and _hash_file(cp)==c['sha256'];bindings.append(c)
    (out/'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-15',target_venue='IEEE TMC',policy_version='tmc_review_policy_v3_20260621',
        artifact_run_id=str(out.relative_to(ROOT)),original_training_source=original['git_commit'],training_added=0,original_training_episodes=384,
        cells=len(rows),checkpoints=bindings,events_equal_cells=len(rows),summary={k:statistics.mean(r[k] for r in rows) for k in ['reward','completed_requests','completed_nodes','model_transfer_link_frames','state_transfer_link_frames']},
        events_sha256=_hash_file(out/'events.jsonl'),evidence_level='dev exact frozen legacy replay; paper E0_UNAVAILABLE',
        scope='strong historical controller reference; same raw physics/protocol and original128x3 budget, no new updates; distinct old controller contract/credit, not claimed as new training')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
