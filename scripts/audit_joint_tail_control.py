"""Read-only producer/consumer, raw-receipt, replay and fixed-weight prediction audit."""
from collections import defaultdict
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_joint_tail_control import make_agent,make_env,tensor_hash
from scripts.train_joint_parameterized_control import episode
from scripts.launch_eight_vehicle_joint_experiment import _hash_file


def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
def avg(values):return statistics.mean(list(values))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    a.roots=[r.resolve() for r in a.roots]
    root=a.roots[0];plan=None if a.smoke else json.loads((root/'comparison_plan.json').read_text())
    paths=[p for r in a.roots for p in r.glob('*/*/manifest.json')]
    if a.smoke:paths=[p for r in a.roots for p in r.glob('*/manifest.json')]
    assert paths
    manifests=[json.loads(p.read_text()) for p in paths]
    if plan:
        assert sorted(m['agent'] for m in manifests)==sorted(plan['primary_agents']+plan['support_agents'])
    dev=[];training=[];probes=[];reproductions=[];predict=[];checkpoints=events_count=receipts_count=0;initial={}
    for path,m in zip(paths,manifests):
        assert m['status']=='complete'
        if plan:assert not m['git_dirty'] and m['git_commit']==plan['source_commit'] and m['episodes']==plan['episodes'] and m['seeds']==plan['seeds']
        prepared=Path(m['prepared_protocol']);assert _hash_file(prepared)==m['prepared_manifest_sha256']
        protocol=json.loads(prepared.read_text())
        if plan:
            assert m['prepared_manifest_sha256']==plan['prepared_manifest_sha256']
            assert m['prediction_sha256']==plan['predictor_sha256']
            assert m['primary_panels']==plan['primary_panels']==protocol['primary_panels']==dict(base=.5,tight=.5)
            assert m['return_gamma']==(.99 if m['agent']=='sa_discounted' else 1.)
        assert _hash_file(prepared.parent/'protocol.pt')==protocol['protocol_sha256']
        assert _hash_file(prepared.parent/'predictor.pt')==m['prediction_sha256']==protocol['predictor_sha256']
        if plan:assert not protocol['git_dirty'] and protocol['git_commit']==plan['source_commit']
        assert not set(protocol['predictor']['fit_offsets']) & set(protocol['predictor']['validation_offsets'])
        intervals=protocol['source_splits']
        assert set(protocol['predictor']['fit_offsets'])=={w['frame_offset'] for w in intervals['train'][:16]}
        assert set(protocol['predictor']['validation_offsets'])=={w['frame_offset'] for w in intervals['train'][16:]}
        for left in intervals['train']:
            for right in intervals['dev']:
                assert left['source_segment_id']!=right['source_segment_id'] or left['time_end']<right['time_start'] or right['time_end']<left['time_start']
        data=torch.load(prepared.parent/'protocol.pt',weights_only=False)
        data['predictor']=torch.load(prepared.parent/'predictor.pt',weights_only=False)
        from src.envs.core.joint_training_scenarios import public_scenarios
        assert public_scenarios(data['scenarios'])==protocol['scenarios']
        assert _hash_file(Path(protocol['source_manifest']))==protocol['source_manifest_sha256']
        windows={w['scenario_id']:w for w in data['scenarios']['dev']}
        assert len(windows)==16
        assert all(sum(w['panel']==panel for w in windows.values())==8 for panel in ('base','tight'))
        rows=read(path.parent/'dev.jsonl');tr=read(path.parent/'training.jsonl');pr=read(path.parent/'train_probe.jsonl')
        assert len(tr)==m['episodes']*len(m['seeds'])
        assert len(rows)==len(m['seeds'])*(m['episodes']//m['eval_every'])*16
        assert len(pr)==len(m['seeds'])*(m['episodes']//m['eval_every'])*4
        for seed in m['seeds']:
            selected=[r for r in tr if r['seed']==seed]
            assert [r['scenario_id'] for r in selected]==[data['scenarios']['train'][(ep-1)%40]['scenario_id'] for ep in range(1,m['episodes']+1)]
        final={(r['seed'],r['scenario_id']):r for r in rows if r['episode']==m['episodes']}
        grouped=defaultdict(list)
        for e in read(path.parent/'events.jsonl'):grouped[(e['seed'],e['scenario_id'])].append(e)
        assert grouped.keys()==final.keys()
        for key,es in grouped.items():
            row=final[key];reward=0.
            receipts=[]
            for e in es:
                w=windows[key[1]];assert e['time_index']==w['frames'][e['frame_index']]['time_index']
                if 'frame_cost_receipt' in e:
                    receipt=e['frame_cost_receipt'];components=receipt['reward_components'];reward+=sum(components.values())
                    receipts.append(receipt)
                    assert abs(sum(receipt['request_rewards'].values())-sum(components.values()))<1e-8
                    assert all(0<=used<=1 for used in receipt['link_usage_by_rsu'].values());receipts_count+=1
                vid=e['vehicle_id'];grants=e['link_reservation_grants']
                for transfer in e['state_transfers']:
                    if transfer['link_frames'] and transfer['target'] in grants:
                        assert grants[transfer['target']]['vehicle_id']==vid and grants[transfer['target']]['kind']=='state'
                for target,kind,used in [(e['control']['cache_target'],'model',e['cache_request']['link_slots_consumed']),
                                        (e['target_rsu_id'],'input',e['input_transfer_link_frames']),
                                        (e['output_source_rsu_id'],'output',e['output_transfer_link_frames'])]:
                    if used and target in grants:
                        assert grants[target]['vehicle_id']==vid and grants[target]['kind']==kind
            assert abs(reward-row['reward'])<1e-8
            assert sum(r['completed'] for r in row['per_request'])==row['completed_requests']
            assert sum(r['completed_nodes'] for r in row['per_request'])==row['completed_nodes']
            assert all(c['occupied_bytes']<=c['capacity_bytes'] for c in row['model_cache'].values())
            assert sum(e['output_transfer_link_frames'] for e in es)==row['output_transfer_link_frames']
            assert sum(e['state_transfer_link_frames'] for e in es)==row['state_transfer_link_frames']
            assert sum(e['input_transfer_link_frames'] for e in es)==row['input_transfer_link_frames']
            assert sum(r['model_transfer_link_frames'] for r in receipts)==row['model_transfer_link_frames']
            assert sum(r['shared_state_transmitted_bytes'] for r in receipts)==row['shared_state_transmitted_bytes']
            assert sum(e['status'] in ('node_completed','vehicle_fallback_completed') for e in es)==row['completed_nodes']
            events_count+=len(es)
        for c in m['checkpoints']:
            cp=path.parent/c['path'];assert _hash_file(cp)==c['sha256'];saved=torch.load(cp,weights_only=False)
            fresh=make_agent(m['agent'],c['seed'],m['episodes']);assert tensor_hash(fresh)==c['initial_tensor_sha256']
            agent=type(fresh).from_checkpoint(cp);assert tensor_hash(agent)==c['trained_tensor_sha256']
            assert any(not torch.equal(v,agent.state_dict()[k]) for k,v in fresh.state_dict().items() if k.startswith('heads.'))
            initial[(m['agent'],c['seed'])]=c['initial_tensor_sha256'];checkpoints+=1
            assert agent.return_gamma==m['return_gamma']
            if c['episode']!=m['episodes']:continue
            for w in data['scenarios']['dev'][:2]:
                env=make_env(w,data,protocol);row,rolls=episode(env,agent,deterministic=True)
                original=[{k:v for k,v in e.items() if k not in ('agent','seed','episode','scenario_id','panel','frame_offset')} for e in grouped[(c['seed'],w['scenario_id'])]]
                assert json.loads(json.dumps(env.events))==original
                assert abs(row['reward']-final[(c['seed'],w['scenario_id'])]['reward'])<1e-8
                if agent.return_gamma==1.:assert abs(float(rolls[0]['returns'][0])*10-row['reward'])<1e-5
                reproductions.append(dict(agent=m['agent'],seed=c['seed'],scenario_id=w['scenario_id'],events_equal=True))
            if m['agent']=='sa_ghmappo' and not a.smoke:
                before=tensor_hash(agent);agent.prediction_enabled=False
                for w in data['scenarios']['dev']:
                    off,_=episode(make_env(w,data,protocol),agent,deterministic=True)
                    on=final[(c['seed'],w['scenario_id'])]
                    predict.append(dict(seed=c['seed'],scenario_id=w['scenario_id'],panel=w['panel'],reward_on=on['reward'],reward_off=off['reward'],delta_reward=on['reward']-off['reward'],delta_completed=on['completed_requests']-off['completed_requests']))
                assert tensor_hash(agent)==before and _hash_file(cp)==c['sha256']
        for row in rows:
            row['cache_evictions']=sum(c['eviction_count'] for c in row['model_cache'].values())
            row['cache_transferred_bytes']=sum(c['transferred_bytes'] for c in row['model_cache'].values())
        dev+=rows;training+=tr;probes+=pr
        print(json.dumps(dict(agent=m['agent'],raw_audited=True,checkpoints_checked=checkpoints,reproductions=len(reproductions))),flush=True)
    if plan:
        for seed in plan['seeds']:
            assert initial[('sa_ghmappo',seed)]==initial[('sa_no_prediction',seed)]==initial[('sa_discounted',seed)]==initial[('sa_no_surrogate',seed)]
        assert len({m['prepared_manifest_sha256'] for m in manifests})==1
    summary={};curves={};paired={}
    for m in manifests:
        name=m['agent'];rs=[r for r in dev if r['agent']==name and r['episode']==m['episodes']]
        metrics=['reward','completed_requests','completed_nodes','model_transfer_link_frames','state_transfer_link_frames','service_target_change_count',
                 'input_transfer_link_frames','output_transfer_link_frames','shared_state_transmitted_bytes','cache_evictions','cache_transferred_bytes']
        summary[name]={k:avg(r[k] for r in rs) for k in metrics}
        summary[name]['by_panel']={panel:{k:avg(r[k] for r in rs if r['panel']==panel) for k in metrics} for panel in ('base','tight')}
        summary[name]['reward_by_seed']=[avg(r['reward'] for r in rs if r['seed']==seed) for seed in m['seeds']]
        summary[name]['reward_seed_std']=statistics.stdev(summary[name]['reward_by_seed']) if len(m['seeds'])>1 else None
        summary[name]['mechanisms']={k:avg(r['mechanisms'][k] for r in rs) for k in rs[0]['mechanisms']}
        curves[name]={}
        for ep in sorted({r['episode'] for r in dev if r['agent']==name}):
            values=[r for r in dev if r['agent']==name and r['episode']==ep];probe=[r for r in probes if r['agent']==name and r['episode']==ep]
            updates=[r['losses'] for r in training if r['agent']==name and ep-m['eval_every']<r['episode']<=ep and r['losses']]
            curves[name][str(ep)]=dict(reward=avg(r['reward'] for r in values),completed=avg(r['completed_requests'] for r in values),
                probe_reward=avg(r['reward'] for r in probe),probe_completed=avg(r['completed_requests'] for r in probe),
                by_seed={str(seed):dict(reward=avg(r['reward'] for r in values if r['seed']==seed),completed=avg(r['completed_requests'] for r in values if r['seed']==seed)) for seed in m['seeds']},
                updates={key:avg(u[key] for u in updates if key in u) for key in ('gradient_steps','approx_kl','kl_early_stop','normalized_entropy','q_mse') if any(key in u for u in updates)})
    if plan:
        full={(r['seed'],r['scenario_id']):r for r in dev if r['agent']=='sa_ghmappo' and r['episode']==plan['episodes']}
        for name in summary:
            if name=='sa_ghmappo':continue
            other={(r['seed'],r['scenario_id']):r for r in dev if r['agent']==name and r['episode']==plan['episodes']}
            ds=[full[k]['reward']-other[k]['reward'] for k in full]
            paired[name]=dict(reward=avg(ds),completed=avg(full[k]['completed_requests']-other[k]['completed_requests'] for k in full),wins=sum(d>1e-9 for d in ds),losses=sum(d< -1e-9 for d in ds))
    destination=(root if not a.smoke else ROOT/'artifacts/joint_tail_preflight_audit')
    destination.mkdir(exist_ok=True)
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-15',target_venue='IEEE TMC',
        policy_version='tmc_review_policy_v3_20260621',source_commit=manifests[0]['git_commit'],
        audit_source_sha256=_hash_file(Path(__file__)),
        artifact_run_id=[str(p.parent.relative_to(ROOT)) for p in paths],evidence_level='development raw/replayed; paper E0_UNAVAILABLE',
        counts=dict(checkpoints=checkpoints,events=events_count,receipts=receipts_count,reproductions=len(reproductions)),
        summary=summary,curves=curves,paired=paired,prediction_probe=predict,reproductions=reproductions,
        statistical_scope='descriptive; base/tight and cohorts share raw intervals; no significance/independent-cluster claim',paper_claim_ready=False)
    (destination/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(destination/'audit.json'),counts=report['counts'],summary=summary,paired=paired),indent=2))


if __name__=='__main__':main()
