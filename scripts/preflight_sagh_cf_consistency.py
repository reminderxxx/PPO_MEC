"""One fixed C/T development comparison; 5184 total real environment calls max."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
import preflight_sagh_execution_repair as pre
import run_fair_reward_training as fair

SOURCE=Path(__file__).resolve().parents[1]
BASE='9cb156afacc6123367c31db99714350ab5953ba2'
VARIANTS=('C','T')
SEEDS=(7,13)
STEPS_PER_CELL=768
TRAIN_CAP=3072
VALIDATION_CAP=2112
TOTAL_CAP=5184
CONTRACTS={'C':'none_v1','T':'native_positive_executed_action_v1'}

class InteractionBudget:
    """Charge before every core env.step, including failed calls. Never refund."""
    def __init__(self,stream):
        self.stream=stream;self.phase=None;self.cell=None
        self.counts={'train':0,'validation':0};self.per_cell={};self.total=0
    def charge(self):
        if self.phase not in self.counts or self.cell is None:
            raise RuntimeError('Environment call outside a budgeted episode')
        counts=self.per_cell.setdefault(self.cell,{'train':0,'validation':0})
        if self.total>=TOTAL_CAP or self.counts[self.phase]>=(TRAIN_CAP if self.phase=='train' else VALIDATION_CAP):
            raise RuntimeError('Global real environment hard cap reached')
        if counts[self.phase]>=(STEPS_PER_CELL if self.phase=='train' else 528):
            raise RuntimeError('Cell real environment hard cap reached')
        self.total+=1;self.counts[self.phase]+=1;counts[self.phase]+=1
        self.stream.write(json.dumps({'ordinal':self.total,'cell':self.cell,'phase':self.phase})+'\n');self.stream.flush()
    def snapshot(self):return dict(total=self.total,counts=self.counts,per_cell=self.per_cell,charged_before_step=True,refunds=0)

def execute(parent,out):
    if out.exists():raise FileExistsError('Create-only C/T run; no resume/retry')
    if parent==out or parent in out.parents:raise ValueError('Cannot write within old run')
    if subprocess.check_output(['git','diff','HEAD','--','src','scripts','tests'],cwd=SOURCE):raise ValueError('Commit matching-tested code first')
    if subprocess.check_output(['git','diff',BASE,'--','src/envs','src/metrics','configs','src/encoders'],cwd=SOURCE):raise ValueError('Reward/environment/encoder changed')
    plan,tr,va=pre.plan_from_parent(parent)
    for key,identity in plan['data_files'].items():
        p=Path(plan[key])
        if p.stat().st_size!=identity['size'] or fair.sha(p)!=identity['sha256']:raise ValueError('Frozen real data changed')
    os.nice(max(0,19-os.nice(0)))
    guard=pre.resource_guard();fair.load_api(SOURCE)
    out.mkdir(parents=True)
    pre.write_json(out/'execution_receipt.json',dict(scope='non_formal_cf_consistency_ct_only',command=sys.argv,
        source_root=str(SOURCE),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        base_commit=BASE,parent_plan_sha256=plan['plan_sha256'],variants=VARIANTS,seeds=SEEDS,contracts=CONTRACTS,
        train_cap=TRAIN_CAP,validation_cap=VALIDATION_CAP,total_environment_cap=TOTAL_CAP,steps_per_cell=STEPS_PER_CELL,
        learner_opportunities_per_cell=3,checkpoints_per_cell=4,train_windows=tr,validation_windows=va,data_files=plan['data_files'],
        only_between_arm_change='counterfactual_consistency_contract',auxiliary_action_contract='executed_marginal_v1',
        denominator='original_eligible_rows_including_paused',native_advantage='normalized_native_GAE_before_mechanism_credits',
        rule='pause iff frozen_native_advantage>0 and target!=current_final_executed_action',
        reward_changes=False,auxiliary_coefficient_changes=False,critic_ablation=False,formal_training_started=False,
        old_checkpoint_loads=0,original_curve_changes=False,automatic_retry=False,
        runtime=dict(cpu_threads=1,nice=19,concurrent_cells=1,yield_to_foreign_training=True),
        source_sha256={str(p.relative_to(SOURCE)):fair.sha(p) for p in sorted(SOURCE.rglob('*.py')) if p.parts[len(SOURCE.parts)] in ('src','scripts','tests')}))
    budget=None;original_step=fair.VecWorkflowCoreEnv.step;original_episode=fair.make_episode
    try:
        train=[];validation=[]
        for records,bundles in [(tr,train),(va,validation)]:
            for record in records:
                bundle=fair.window_bundle(SOURCE,plan,record['frame_offset'])
                if fair.window_identity(bundle)!={k:record[k] for k in fair.window_identity(bundle)}:raise ValueError('Window identity mismatch')
                bundle.rsu_metadata['window_class']=record['window_class'];bundles.append(bundle)
        workflows=fair.build_workflows(workflow_csv_path=plan['workflow_path'],max_workflows=plan['max_workflows'],workflow_selector='ordered',min_tasks=5,max_tasks=20,random_seed=7)
        if [w.workflow_id for w in workflows]!=plan['workflow_ids']:raise ValueError('Workflow identity mismatch')
        with (out/'interaction_ledger.jsonl').open('x') as stream:
            budget=InteractionBudget(stream)
            def charged_step(env,*args,**kwargs):budget.charge();return original_step(env,*args,**kwargs)
            def budgeted_episode(source,p,agent,bundle,workflow,seed,max_steps,metadata,training):
                if budget.phase is not None:raise ValueError('Nested environment episode forbidden')
                budget.phase='train' if training else 'validation'
                before=budget.total
                try:
                    result=original_episode(source,p,agent,bundle,workflow,seed,max_steps,metadata,training)
                    if budget.total-before!=len(result[1]):raise ValueError('Unrecorded environment calls')
                    return result
                finally:budget.phase=None
            fair.VecWorkflowCoreEnv.step=charged_step;fair.make_episode=budgeted_episode
            for variant in VARIANTS:
                arm=out/'variants'/variant;arm.mkdir(parents=True)
                def configured(current_plan,name,seed):
                    cfg=copy.deepcopy(current_plan['agent_configs'][name][str(seed)])
                    cfg.update(execution_mapping_contract='shared_train_eval_v1',objective_diagnostics_enabled=True,
                        auxiliary_action_contract='executed_marginal_v1',counterfactual_consistency_contract=CONTRACTS[variant])
                    return fair.fresh_agent(name,cfg,seed),cfg
                pre.configured_agent=configured
                for seed in SEEDS:
                    budget.cell=f'{variant}/seed{seed}'
                    pre.run_cell(arm,plan,'sa_ghmappo',seed,train,validation,workflows,guard)
        cells=[json.loads(p.read_text()) for p in out.glob('variants/*/cells/sa_ghmappo/seed*/complete.json')]
        if len(cells)!=4 or sum(x['training_steps'] for x in cells)!=TRAIN_CAP:raise ValueError('Four fresh cells/train cap failed')
        validation_steps=sum(x['validation_steps'] for x in cells)
        if budget.counts!={'train':TRAIN_CAP,'validation':validation_steps} or budget.total> TOTAL_CAP:raise ValueError('Budget ledger mismatch')
        pre.write_json(out/'completion.json',dict(status='PREFLIGHT_COMPLETED',cells=4,fresh_cells=4,training_steps=TRAIN_CAP,
            validation_steps=validation_steps,total_environment_steps=budget.total,learner_opportunities=12,checkpoints=16,
            budget=budget.snapshot(),formal_training_started=False,critic_ablation=False,reward_changes=False,auxiliary_coefficient_changes=False))
    except Exception as error:
        pre.write_json(out/'failure.json',dict(status='PREFLIGHT_FAILED',error_type=type(error).__name__,error=str(error),
            budget=budget.snapshot() if budget else None,automatic_retry=False))
        raise
    finally:fair.VecWorkflowCoreEnv.step=original_step;fair.make_episode=original_episode

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--execute-preflight',action='store_true',required=True)
    args=p.parse_args();execute(args.parent_root.resolve(),args.output_root.resolve())
