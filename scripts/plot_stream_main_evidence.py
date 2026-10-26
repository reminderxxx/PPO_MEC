"""Plot observed seed means and signed module effects, without formal CI claims."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full_cohort_runs',type=Path,nargs='+',required=True)
    p.add_argument('--full_cohort_audit',type=Path,required=True)
    p.add_argument('--ablation_audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for root in a.full_cohort_runs for s in (root/'results.jsonl').read_text().splitlines()]
    wide=json.loads(a.full_cohort_audit.read_text());small=json.loads(a.ablation_audit.read_text())
    assert len(rows)==72 and wide['verified_evaluation_rows']==72
    a.output.mkdir(parents=True,exist_ok=False)
    colors=['#66788a','#1768ac','#d55e00']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),gridspec_kw={'width_ratios':[1,1,1.25]})
    seeds=[7,17,27]
    for ax,field,label in [(axes[0],'reward','Native return'),
            (axes[1],'switches','Execution-site changes')]:
        for j,seed in enumerate(seeds):
            means=[]
            for variant in ['original','relocation']:
                selected=[r for r in rows if r['variant']==variant and r['seed']==seed]
                value=[r['reward'] if field=='reward' else r['counts']['service_target_changed'] for r in selected]
                means.append(float(np.mean(value)))
            ax.plot([0,1],means,'o-',color=colors[j],alpha=.9,label=f'Seed {seed}',markersize=6)
        ax.set_xticks([0,1],['Original\n+ same training','Relocation\n+ same training'])
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_xlim(-.2,1.2)
    axes[0].axhline(wide['rule_mean'],color='#333333',linestyle='--',linewidth=1.2,label='Frozen gain_prefetch')
    axes[0].set_title('A  All 12 training conditions')
    axes[0].legend(frameon=False,fontsize=8,loc='best')
    axes[1].set_title('B  Physical mechanism')
    names=['no_dag_edges','no_prediction','no_surrogate','no_optional_cache']
    labels=['DAG edges','Prediction','Surrogate','Optional cache']
    methods=small['methods'];full=methods['relocation']['final_per_seed'];values=[]
    for i,name in enumerate(names):
        effects=[full[str(s)]-methods[name]['final_per_seed'][str(s)] for s in seeds]
        values.append(effects)
        axes[2].bar(i,np.mean(effects),color='#a7c7e7',width=.6,zorder=2)
        for j,effect in enumerate(effects):
            axes[2].scatter(i+(j-1)*.12,effect,color=colors[j],s=24,zorder=3)
    axes[2].axhline(0,color='#333333',linewidth=1)
    axes[2].set_xticks(range(4),labels,rotation=22,ha='right')
    axes[2].set_ylabel('Full candidate minus module deletion')
    axes[2].set_title('C  Three-condition warm-start ablations')
    axes[2].grid(axis='y',alpha=.2,zorder=0)
    fig.suptitle('Main-method development evidence',fontsize=16,fontweight='bold',y=.99)
    fig.text(.5,.01,'Three training-time sessions; dots are seed means, not independent time clusters. '
        'Fixed-final checkpoints; all signs retained. No formal or holdout inference.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.08,1,.93])
    for extension in ['png','pdf','svg']:
        fig.savefig(a.output/f'main_method_evidence.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (a.output/'plot_data.json').write_text(json.dumps(dict(full_cohort=wide['methods'],
        module_effects=dict(zip(names,values)),caption='Seed variability only; no formal confidence interval'),indent=2)+'\n')
    print(str(a.output/'main_method_evidence.png'))


if __name__=='__main__':main()
