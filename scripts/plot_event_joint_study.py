"""Publication figures from all fixed checkpoints and all confirmation rows."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_persistent_workflow_learning import dump,sha
from scripts.analyze_event_joint_study import paired_hierarchical_bootstrap

LABELS={'event_set_ppo':'Event PPO','event_graph_ppo':'Event PPO + graph','primitive_ppo':'Step PPO',
 'no_prefetch':'No prefetch','shortest_frontier':'Shortest frontier','ready_demand':'Ready demand','min_transfer':'Min transfer','eviction_cost':'Eviction cost'}
COLORS={'event_set_ppo':'#0072B2','event_graph_ppo':'#7B4EA3','primitive_ppo':'#D55E00'}


def save(fig,path,receipt):
    for ext in ['png','pdf','svg']:
        target=path.with_suffix('.'+ext);fig.savefig(target,dpi=300,bbox_inches='tight');receipt.append(dict(path=str(target),sha256=sha(target)))
    plt.close(fig)


def plot(analysis_dir,out):
    out.mkdir(parents=True);a=json.loads((analysis_dir/'analysis.json').read_text());learning=json.loads((analysis_dir/'learning_rows.json').read_text());conf=json.loads((analysis_dir/'confirmation_rows.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','xtick.direction':'out','ytick.direction':'out'})
    receipt=[];fig,axs=plt.subplots(1,2,figsize=(9.2,3.4));seeds=[7,13,29];episodes=list(range(0,257,16))
    for method in ['event_set_ppo','event_graph_ppo','primitive_ppo']:
        values=np.array([[np.mean([r['reward'] for r in learning if r['method']==method and r['seed']==s and r['episode']==e]) for e in episodes] for s in seeds])
        for ax,data in zip(axs,[values,values-values[:,0:1]]):
            mean=data.mean(0);sd=data.std(0,ddof=1);ax.plot(episodes,mean,label=LABELS[method],color=COLORS[method],lw=1.8)
            ax.fill_between(episodes,mean-sd,mean+sd,color=COLORS[method],alpha=.14,lw=0)
    root=analysis_dir.parent
    dev_file=root/'dev_baselines_v2/shortest_frontier/rows.json'
    if dev_file.exists():
        raw=json.loads(dev_file.read_text());reference=np.mean([x['reward'] for x in raw]);axs[0].axhline(reference,color='#009E73',ls='--',lw=1.3,label='Shortest-frontier heuristic')
    axs[1].axhline(0,color='0.55',lw=.7)
    for i,ax in enumerate(axs):
        ax.set_xlabel('Training episodes');ax.set_xlim(0,256);ax.set_xticks([0,64,128,192,256]);ax.grid(axis='y',alpha=.18,lw=.6)
        ax.set_title(['(a) Fixed development evaluation','(b) Gain over the same policy at initialization'][i],loc='left',fontsize=9.5)
    axs[0].set_ylabel('Undiscounted business return');axs[1].set_ylabel('Return gain from episode 0')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.51,1.05),fontsize=8)
    fig.tight_layout(rect=[0,0,1,.93]);save(fig,out/'reward_convergence',receipt)
    # Paired effects include every baseline, with a fixed zero reference.
    comps=[x for x in a['comparisons'] if x['metric']=='reward'];fig,ax=plt.subplots(figsize=(6.7,3.6));y=np.arange(len(comps))
    for i,x in enumerate(comps):
        lo,hi=x['bca95'];mean=x['estimate'];color='#0072B2' if x['positive_corrected_evidence'] else '#777777'
        ax.errorbar(mean,i,xerr=[[max(mean-lo,0.)],[max(hi-mean,0.)]],fmt='o',color=color,capsize=3,ms=4,lw=1.2)
    ax.axvline(0,color='0.25',lw=.9);ax.set_yticks(y);ax.set_yticklabels([LABELS[x['comparator']] for x in comps]);ax.invert_yaxis()
    ax.set_xlabel('Event PPO return minus comparator return (95% BCa interval)');ax.grid(axis='x',alpha=.18)
    ax.set_title(f'Primary cohort: {a["effective_primary_configurations"]} effective input configurations',loc='left',fontsize=10);fig.tight_layout();save(fig,out/'confirmation_paired_effects',receipt)
    comps=[x for x in a['supporting']['comparisons'] if x['metric']=='reward'];fig,ax=plt.subplots(figsize=(6.7,3.6))
    for i,x in enumerate(comps):
        lo,hi=x['bca95'];mean=x['estimate'];color='#0072B2' if x['positive_corrected_evidence'] else '#777777'
        ax.errorbar(mean,i,xerr=[[max(mean-lo,0.)],[max(hi-mean,0.)]],fmt='o',color=color,capsize=3,ms=4,lw=1.2)
    ax.axvline(0,color='0.25',lw=.9);ax.set_yticks(np.arange(len(comps)));ax.set_yticklabels([LABELS[x['comparator']] for x in comps]);ax.invert_yaxis()
    ax.set_xlabel('Event PPO return minus comparator return (95% BCa interval)');ax.grid(axis='x',alpha=.18)
    ax.set_title('Complete 36-composition matrix in 12 outer time windows',loc='left',fontsize=10);fig.tight_layout();save(fig,out/'supporting_paired_effects',receipt)
    support=json.loads((analysis_dir/'supporting_confirmation_rows.json').read_text())
    # Real service and resource metrics; no rescaling of rewards.
    methods=['event_set_ppo','event_graph_ppo','primitive_ppo','no_prefetch','shortest_frontier','ready_demand','min_transfer','eviction_cost']
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.8));metrics=['completed_requests','waiting_seconds','transfer_mb'];titles=['Completed requests / 8','Pending-time area (request-seconds)','Model transfer (MB)']
    for ax,metric,title in zip(axs,metrics,titles):
        means=[a['supporting']['table'][m][metric] for m in methods];colors=[COLORS.get(m,'#8F9A9D') for m in methods]
        ax.barh(np.arange(len(methods)),means,color=colors,height=.65)
        for i,method in enumerate(methods):
            rr=[r for r in support if r['method']==method];windows=sorted({r['outer_window'] for r in rr});seeds_m=[7,13,29] if method in COLORS else [0]
            lookup={(r['window'],r['seed']):r[metric] for r in rr};cases={w:sorted({r['window'] for r in rr if r['outer_window']==w}) for w in windows}
            matrix=np.array([[[lookup[(c,seed)] for seed in seeds_m] for c in cases[w]] for w in windows]);ci=paired_hierarchical_bootstrap(matrix)['bca95']
            ax.errorbar(means[i],i,xerr=[[max(means[i]-ci[0],0.)],[max(ci[1]-means[i],0.)]],fmt='none',ecolor='0.2',capsize=2,lw=.8)
        ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_yticks(np.arange(len(methods)));ax.set_yticklabels([LABELS[m] for m in methods] if ax is axs[0] else [])
    axs[0].set_xlim(0,8);fig.tight_layout();save(fig,out/'supporting_service_cost',receipt)
    dump(out/'figure_manifest.json',dict(files=receipt,source_analysis_sha256=sha(analysis_dir/'analysis.json'),
        learning='All 17 prespecified checkpoint evaluations; no smoothing; lines are means of 3 seed means across 8 dev windows; shaded regions seed SD',
        paired='All 7 comparisons per cohort; primary uses five effective input configurations and supporting uses 12 time windows / 3 workloads / 3 seeds; BCa95 and a joint 56-test Holm family; blue only when positive CI and corrected sign test <.05',
        service='All 36 supporting compositions; three workloads per outer window; BCa95 hierarchy window/workload/seed; deterministic rules are not independent repeated seeds',
        selection='Primary method and final checkpoint frozen before confirmation; no successful-window restriction',formal=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot(Path(a.analysis),Path(a.output))
