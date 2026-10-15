"""Read-only fixed-budget greedy train/dev and stability-treatment diagnostics."""
from pathlib import Path
from datetime import datetime, timezone
import json
import statistics
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]

def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def avg(values):
    values=list(values)
    return statistics.mean(values) if values else None

def main():
    experiment=sys.argv[1] if len(sys.argv)>1 else 'joint_scheduled_control'
    root=ROOT/'artifacts'/experiment;plan=json.loads((root/'comparison_plan.json').read_text())
    ap=root/'comparison_amendment_1.json';amendment=json.loads(ap.read_text()) if ap.exists() else None
    groups=dict(plan['groups'])
    if amendment:groups.update(amendment['replacement_groups'])
    summary={};checkpoints={};all_manifest=[]
    for group,original_names in groups.items():
        label=lambda name:amendment['aliases'].get(name,name) if amendment and group==amendment['original_group'] else name
        names=[label(name) for name in original_names]
        per_agent={name:dict(training=[],dev=[],probe=[],representation=[]) for name in names}
        for seed in plan['seeds']:
            paths=list((root/group/f'seed{seed}').glob('*/manifest.json'));assert len(paths)==1
            path=paths[0];m=json.loads(path.read_text());all_manifest.append(str(path.relative_to(ROOT)))
            assert m['status']=='complete' and not m['git_dirty'] and m['git_commit']==(amendment['source_commit'] if amendment and group in amendment['replacement_groups'] else plan['source_commit'])
            assert m['agents']==original_names and m['seeds']==[seed]
            for kind,filename in [('training','training.jsonl'),('dev','dev.jsonl'),('probe','train_probe.jsonl'),('representation','representation_diagnostics.jsonl')]:
                for row in read(path.parent/filename):
                    name=label(row['agent'])
                    if name in per_agent:per_agent[name][kind].append(dict(row,agent=name))
            for c in m['checkpoints']:checkpoints[(label(c['agent']),seed,c['episode'])]=(path.parent/c['path'],c)
        for name,data in per_agent.items():
            curves={};training=data['training'];updates=[r for r in training if r['losses'] is not None]
            assert len(training)==plan['episodes']*len(plan['seeds'])
            for ep in (32,64,96,128):
                dev=[r for r in data['dev'] if r['episode']==ep];probe=[r for r in data['probe'] if r['episode']==ep]
                assert len(dev)==24 and len(probe)==12
                curves[str(ep)]=dict(dev_reward=avg(r['reward'] for r in dev),dev_completed=avg(r['completed_requests'] for r in dev),train_probe_reward=avg(r['reward'] for r in probe),train_probe_completed=avg(r['completed_requests'] for r in probe),graph_initial_saturation=avg(r['first_dev_initial_graph_tanh_saturation'] for r in data['representation'] if r['episode']==ep))
                curves[str(ep)].update(dev_completed_nodes=avg(r['completed_nodes'] for r in dev),
                    dev_model_link_frames=avg(r['model_transfer_link_frames'] for r in dev),
                    dev_state_link_frames=avg(r['state_transfer_link_frames'] for r in dev),
                    dev_service_changes=avg(r['service_target_change_count'] for r in dev),
                    dev_prepare_commands=avg(r['mechanisms']['prepare_commands'] for r in dev),
                    dev_output_queue_frames=avg(r['status_counts'].get('output_link_queue',0) for r in dev),
                    dev_compute_queue_frames=avg(r['status_counts'].get('compute_queue',0) for r in dev),
                    dev_evictions=avg(sum(c['eviction_count'] for c in r['model_cache'].values()) for r in dev))
            stats={}
            for key in ('gradient_steps','learning_rate','approx_kl','clip_fraction','normalized_entropy','q_mse','kl_early_stop','value_loss','surrogate_loss','epsilon','replay_size'):
                values=[r['losses'][key] for r in updates if key in r['losses']]
                stats[key]=dict(mean=avg(values),maximum=max(values) if values else None,
                    first8_mean=avg(r['losses'][key] for r in updates if r['episode']<=32 and key in r['losses']),last8_mean=avg(r['losses'][key] for r in updates if r['episode']>96 and key in r['losses']))
            per_seed={str(seed):dict(first20_stochastic_reward=avg(r['reward'] for r in training if r['seed']==seed and r['episode']<=20),last20_stochastic_reward=avg(r['reward'] for r in training if r['seed']==seed and r['episode']>108)) for seed in plan['seeds']}
            summary[name]=dict(curves=curves,update_statistics=stats,stochastic_training_by_seed=per_seed,
                final_minus32_dev_reward=curves['128']['dev_reward']-curves['32']['dev_reward'],final_minus32_train_probe_reward=curves['128']['train_probe_reward']-curves['32']['train_probe_reward'])
            previous={(r['seed'],r['frame_offset']):r for r in data['dev'] if r['episode']==96}
            final={(r['seed'],r['frame_offset']):r for r in data['dev'] if r['episode']==128}
            assert previous.keys()==final.keys() and len(final)==24
            late=[]
            for key in sorted(final):
                before=previous[key];after=final[key]
                delta={field:after[field]-before[field] for field in ('reward','completed_requests','completed_nodes','service_target_change_count','model_transfer_link_frames','state_transfer_link_frames')}
                delta.update({field:after['status_counts'].get(field,0)-before['status_counts'].get(field,0) for field in ('compute_queue','output_link_queue')})
                late.append(dict(seed=key[0],frame_offset=key[1],delta=delta))
            summary[name]['final_minus96_paired_dev']=late
            summary[name]['training_mechanism_bands']={}
            for band_label,predicate in [('first32',lambda r:r['episode']<=32),('last32',lambda r:r['episode']>96)]:
                band=[r for r in training if predicate(r)]
                summary[name]['training_mechanism_bands'][band_label]=dict(
                    episodes=len(band),
                    means={field:avg(r[field] for r in band) for field in ('reward','completed_requests','service_target_change_count','model_transfer_link_frames','state_transfer_link_frames')},
                    mechanisms={field:avg(r['mechanisms'][field] for r in band) for field in ('prepare_commands','future_cache_commands','cache_commands')})
    paired=[]
    for seed in plan['seeds']:
        a,ac=checkpoints[('sa_ghmappo',seed,128)];b,bc=checkpoints[('sa_constant_update',seed,128)]
        assert ac['initial_tensor_sha256']==bc['initial_tensor_sha256']
        aa=torch.load(a,weights_only=False)['options'];bb=torch.load(b,weights_only=False)['options']
        assert aa.pop('stability_enabled') and not bb.pop('stability_enabled');assert aa==bb
        paired.append(dict(seed=seed,initial_tensors_equal=True,only_option_difference='stability_enabled'))
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),source_commit=plan['source_commit'],manifest_paths=all_manifest,
        evidence='development diagnostics only; raw artifact integrity/replay audited separately; no holdout or statistical independence claim',summary=summary,paired_stability_identity=paired,
        stability_bundle_final_reward_difference=summary['sa_ghmappo']['curves']['128']['dev_reward']-summary['sa_constant_update']['curves']['128']['dev_reward'],
        limits='fixed four train probes differ from eight dev; descriptive trends do not isolate overfitting, LR versus KL, or individual mechanism causes; no peak checkpoint selection')
    destination=ROOT/'artifacts/joint_coordinated_audit_20261015'/f'{experiment}_training_diagnostics.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(destination),stability_bundle_difference=report['stability_bundle_final_reward_difference'],paired=paired),indent=2))
if __name__=='__main__':main()
