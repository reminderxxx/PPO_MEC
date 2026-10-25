"""Read-only paired analysis; no candidate selection or simulator interaction."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_persistent_workflow_learning import dump,sha

METRICS=('reward','completed_requests','waiting_seconds','transfer_mb')
DIRECTION=dict(reward=1.,completed_requests=1.,waiting_seconds=-1.,transfer_mb=-1.)
NORMAL=NormalDist()


def sign_test(differences,tolerance=1e-7):
    d=np.asarray(differences,dtype=float);positive=int((d>tolerance).sum());negative=int((d < -tolerance).sum());n=positive+negative
    p=min(1.,2.*sum(math.comb(n,k) for k in range(min(positive,negative)+1))/2**n) if n else 1.
    return dict(positive=positive,negative=negative,ties=len(d)-n,p_value=p)


def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.ones(len(pvalues));running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(len(pvalues)-rank)*pvalues[i]));adjusted[i]=running
    return adjusted.tolist()


def paired_hierarchical_bootstrap(differences,replicates=9999,seed=20261026):
    """Outer time windows; paired training seeds resampled within each window."""
    d=np.asarray(differences,dtype=float)
    if d.ndim==2:d=d[:,None,:]
    if d.ndim!=3 or d.shape[0]<2 or min(d.shape[1:])<1 or not np.isfinite(d).all():raise ValueError('Need finite window-by-workload-by-seed pairs')
    w,k,s=d.shape;rng=np.random.default_rng(seed);outer=rng.integers(w,size=(replicates,w));middle=rng.integers(k,size=(replicates,w,k));inner=rng.integers(s,size=(replicates,w,k,s))
    boot=d[outer[:,:,None,None],middle[:,:,:,None],inner].mean(axis=(1,2,3));estimate=float(d.mean());percentile=np.quantile(boot,[.025,.975])
    fraction=(np.sum(boot<estimate)+.5*np.sum(boot==estimate))/replicates
    z0=NORMAL.inv_cdf(float(np.clip(fraction,.5/replicates,1.-.5/replicates)))
    means=d.mean(axis=(1,2));jack=(means.sum()-means)/(w-1);center=jack.mean()-jack
    denominator=6.*float(np.square(center).sum())**1.5;acceleration=float(np.power(center,3).sum())/denominator if denominator>0 else 0.
    adjusted=[]
    for alpha in [.025,.975]:
        z=NORMAL.inv_cdf(alpha);divisor=1.-acceleration*(z0+z)
        q=NORMAL.cdf(z0+(z0+z)/divisor) if abs(divisor)>1e-12 else alpha
        adjusted.append(float(np.clip(q,0.,1.)))
    bca=np.quantile(boot,sorted(adjusted));sd=float(means.std(ddof=1));test=sign_test(means)
    return dict(estimate=estimate,percentile95=percentile.tolist(),bca95=bca.tolist(),outer_windows=w,inner_workloads=k,inner_seeds=s,
        window_mean_differences=means.tolist(),paired_dz=estimate/sd if sd>1e-12 else None,sign_test=test,bootstrap_replicates=replicates,bootstrap_seed=seed)


def collect_learning(root,plan):
    rows=[];training=[];cells=[];summary={}
    for method in plan['methods']:
        for seed in plan['seeds']:
            path=root/'study_v1'/method/f'seed_{seed}'
            completion=json.loads((path/'completion.json').read_text())
            if completion['status']!='complete' or completion['episodes']!=plan['episodes_per_cell']:raise ValueError('Incomplete fixed-budget cell')
            validation=json.loads((path/'validation.json').read_text());episodes=sorted({x['episode'] for x in validation})
            if episodes!=list(range(0,257,16)):raise ValueError('Missing fixed validation checkpoint')
            if len(validation)!=17*8 or any(sorted(x['window'] for x in validation if x['episode']==e)!=list(range(16,24)) for e in episodes):raise ValueError('Incomplete or repeated dev window exposure')
            for row in validation:rows.append(dict(method=method,**row))
            tr=json.loads((path/'training.json').read_text())
            if len(tr)!=256:raise ValueError('Training episode count mismatch')
            for row in tr:training.append(dict(method=method,seed=seed,**row))
            updates=json.loads((path/'updates.json').read_text())
            cells.append(dict(method=method,seed=seed,native_steps=completion['native_steps'],episodes=256,updates=completion['updates'],
                decision_rows=sum(x['rows'] for x in updates),checkpoint=str(path/'checkpoint_256.pt'),checkpoint_sha256=sha(path/'checkpoint_256.pt')))
        group=[r for r in rows if r['method']==method];curve=[]
        for episode in range(0,257,16):
            rr=[r for r in group if r['episode']==episode]
            curve.append(dict(episode=episode,**{m:float(np.mean([x[m] for x in rr])) for m in METRICS}))
        late=[x['reward'] for x in curve if x['episode'] in plan['late_reward_stability']['checkpoints']]
        summary[method]=dict(initial=curve[0],final=curve[-1],learning_reward_gain=curve[-1]['reward']-curve[0]['reward'],curve=curve,
            late_reward_range=max(late)-min(late),late_first_last_gap=abs(late[0]-late[-1]),
            descriptive_late_stability=(max(late)-min(late)<=2. and abs(late[0]-late[-1])<=1.),policy_convergence_proven=False)
    return rows,training,cells,summary


def compare_matrices(matrices,plan):
    comparisons=[];primary=plan['primary_method']
    for method in matrices:
        if method==primary:continue
        for metric in METRICS:
            delta=DIRECTION[metric]*(matrices[primary][metric]-matrices[method][metric])
            result=paired_hierarchical_bootstrap(delta,plan['statistics']['bootstrap_replicates'],plan['statistics']['bootstrap_seed'])
            comparisons.append(dict(primary=primary,comparator=method,metric=metric,positive_direction='primary improves',**result))
    corrected=holm([x['sign_test']['p_value'] for x in comparisons])
    for x,p in zip(comparisons,corrected):x['holm_p_value']=p
    return comparisons


def cohort_statistics(records,plan):
    methods=list(plan['methods'])+plan['causal_rules'];windows=sorted({x.get('outer_window',x['window']) for x in records})
    if len(windows)!=12:raise ValueError('Confirmation requires all twelve outer windows')
    cases=sorted({x['window'] for x in records});case_by_window={w:sorted({x['window'] for x in records if x.get('outer_window',x['window'])==w}) for w in windows}
    if len({len(x) for x in case_by_window.values()})!=1:raise ValueError('Unequal within-window workload counts')
    tables={};matrices={}
    for method in methods:
        rr=[x for x in records if x['method']==method];expected=len(cases)*(3 if method in plan['methods'] else 1)
        if len(rr)!=expected or len({(x['seed'],x['window']) for x in rr})!=expected:raise ValueError('Incomplete or repeated confirmation paired units')
        tables[method]=dict(rows=len(rr),**{m:float(np.mean([x[m] for x in rr])) for m in METRICS});matrices[method]={}
        for metric in METRICS:
            lookup={(x['window'],x['seed']):x[metric] for x in rr}
            matrices[method][metric]=np.array([[[lookup[(c,s if method in plan['methods'] else 0)] for s in plan['seeds']] for c in case_by_window[w]] for w in windows])
    return tables,compare_matrices(matrices,plan),matrices


def analyze(root,out,plan_path):
    out.mkdir(parents=True);plan=json.loads(plan_path.read_text());learning,training,cells,curves=collect_learning(root,plan)
    confirmation=[]
    for path in sorted((root/'confirmation_v1').glob('*/rows.json')):
        for row in json.loads(path.read_text()):confirmation.append(dict(row,result_source=str(path)))
    if not confirmation:raise ValueError('No frozen confirmation results')
    tables,nominal,matrices=cohort_statistics(confirmation,plan)
    diversity=json.loads((root/'input_diversity_readonly_review.json').read_text())
    clusters=diversity['input_effect_projection_classes']
    raw_windows=sorted({x.get('outer_window',x['window']) for x in confirmation})
    grouped={}
    for method,metrics in matrices.items():
        grouped[method]={m:np.stack([d[[raw_windows.index(i) for i in ids]].mean(axis=0) for ids in clusters]) for m,d in metrics.items()}
    comparisons=compare_matrices(grouped,plan)
    for x in comparisons:x['estimand']='Equal weights for effective input-projection classes; original raw-window means and nominal intervals retained separately'
    support=[]
    for path in sorted((root/'supporting_confirmation_v1').glob('*/rows.json')):
        for row in json.loads(path.read_text()):support.append(dict(row,result_source=str(path)))
    if not support:raise ValueError('Prospectively frozen supporting matrix is incomplete')
    supporting_table,supporting_comparisons,_=cohort_statistics(support,plan)
    combined=comparisons+supporting_comparisons
    adjusted=holm([x['sign_test']['p_value'] for x in combined])
    for x,p in zip(combined,adjusted):x['joint_holm_p_value']=p;x['positive_corrected_evidence']=x['bca95'][0]>0 and p<.05
    result=dict(protocol=plan['protocol'],plan_sha256=sha(plan_path),primary_method=plan['primary_method'],learning=curves,confirmation_table=tables,
        comparisons=comparisons,nominal_primary_comparisons=nominal,effective_primary_configurations=len(clusters),
        supporting=dict(table=supporting_table,comparisons=supporting_comparisons,cases=36,outer_time_windows=12),total_training_native_steps=sum(x['native_steps'] for x in cells),cells=cells,
        statistical_unit='Primary effective input configuration; supporting time-window / workload / paired-seed hierarchy',multiplicity_family_size=len(combined),
        fixed_checkpoint=256,formal=False,canonical=False,limitations=plan['limits'])
    dump(out/'analysis.json',result);dump(out/'learning_rows.json',learning);dump(out/'training_rows.json',training);dump(out/'confirmation_rows.json',confirmation);dump(out/'supporting_confirmation_rows.json',support)
    for name,records in [('learning_curves.csv',learning),('confirmation_rows.csv',confirmation),('supporting_confirmation_rows.csv',support)]:
        fields=sorted(set().union(*(r.keys() for r in records)))
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(confirmation_table=tables,training_native_steps=result['total_training_native_steps'],comparisons=len(comparisons))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',default=str(ROOT/'configs/experiment/event_joint_learning_v1.json'));a=p.parse_args();analyze(Path(a.root),Path(a.output),Path(a.plan))
