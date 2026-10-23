"""Fixed 2-contract x 2-seed comparison; no full-training or resume mode."""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import preflight_sagh_execution_repair as pre
import run_fair_reward_training as fair

SOURCE=Path(__file__).resolve().parents[1]
SEEDS=(7,13)
VARIANTS=('proposal_labels_v1','executed_marginal_v1')
STEPS_PER_CELL=768
TRAIN_CAP=3072
VALIDATION_CAP=2112


def execute(parent,out):
    if out.exists():raise FileExistsError('Create-only preflight; no resume/retry')
    if parent==out or parent in out.parents:raise ValueError('Cannot write inside old run')
    if subprocess.check_output(['git','diff','HEAD','--','src','scripts','tests'],cwd=SOURCE):
        raise ValueError('Commit tested source before fixed comparison')
    if subprocess.check_output(['git','diff','495da56','--','src/envs','src/metrics','configs'],cwd=SOURCE):
        raise ValueError('Environment/reward/config bytes changed')
    plan,tr,va=pre.plan_from_parent(parent)
    for key,identity in plan['data_files'].items():
        p=Path(plan[key])
        if p.stat().st_size!=identity['size'] or fair.sha(p)!=identity['sha256']:raise ValueError('Frozen data changed')
    os.nice(max(0,19-os.nice(0)))
    guard=pre.resource_guard();fair.load_api(SOURCE)
    out.mkdir(parents=True)
    pre.write_json(out/'execution_receipt.json',dict(scope='non_formal_auxiliary_alignment_comparison_only',
        command=sys.argv,source_root=str(SOURCE),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        parent_plan_sha256=plan['plan_sha256'],variants=VARIANTS,seeds=SEEDS,train_windows=tr,validation_windows=va,
        train_cap=TRAIN_CAP,validation_cap=VALIDATION_CAP,total_environment_cap=TRAIN_CAP+VALIDATION_CAP,
        steps_per_cell=STEPS_PER_CELL,learner_opportunities_per_cell=3,checkpoints_per_cell=4,
        config_changes=dict(common={'execution_mapping_contract':'shared_train_eval_v1','objective_diagnostics_enabled':True},
            only_between_arm_change='auxiliary_action_contract'),reward_changes=False,auxiliary_weight_changes=False,
        old_checkpoint_loads=0,original_curve_changes=False,formal_training_started=False,automatic_retry=False,
        runtime=dict(cpu_threads=1,nice=19,concurrent_cells=1,yield_to_foreign_training=True),data_files=plan['data_files'],
        source_sha256={str(p.relative_to(SOURCE)):fair.sha(p) for p in sorted(SOURCE.rglob('*.py')) if p.parts[len(SOURCE.parts)] in ('src','scripts','tests')}))
    try:
        train=[];validation=[]
        for records,bundles in [(tr,train),(va,validation)]:
            for record in records:
                bundle=fair.window_bundle(SOURCE,plan,record['frame_offset'])
                if fair.window_identity(bundle)!={k:record[k] for k in fair.window_identity(bundle)}:raise ValueError('Window mismatch')
                bundle.rsu_metadata['window_class']=record['window_class'];bundles.append(bundle)
        workflows=fair.build_workflows(workflow_csv_path=plan['workflow_path'],max_workflows=plan['max_workflows'],
            workflow_selector='ordered',min_tasks=5,max_tasks=20,random_seed=7)
        if [w.workflow_id for w in workflows]!=plan['workflow_ids']:raise ValueError('Workflow mismatch')
        for variant in VARIANTS:
            arm=out/'variants'/variant;arm.mkdir(parents=True)
            def configured(current_plan,name,seed):
                cfg=copy.deepcopy(current_plan['agent_configs'][name][str(seed)])
                cfg.update(execution_mapping_contract='shared_train_eval_v1',objective_diagnostics_enabled=True,auxiliary_action_contract=variant)
                return fair.fresh_agent(name,cfg,seed),cfg
            pre.configured_agent=configured
            for seed in SEEDS:
                pre.run_cell(arm,plan,'sa_ghmappo',seed,train,validation,workflows,guard)
        cells=[json.loads(p.read_text()) for p in out.glob('variants/*/cells/sa_ghmappo/seed*/complete.json')]
        if len(cells)!=4 or sum(x['training_steps'] for x in cells)!=TRAIN_CAP:raise ValueError('Hard train cap/accounting failed')
        validation_steps=sum(x['validation_steps'] for x in cells)
        if validation_steps>VALIDATION_CAP:raise ValueError('Hard validation cap failed')
        pre.write_json(out/'completion.json',dict(status='PREFLIGHT_COMPLETED',cells=4,training_steps=TRAIN_CAP,
            validation_steps=validation_steps,total_environment_steps=TRAIN_CAP+validation_steps,learner_opportunities=12,
            checkpoints=16,fresh_cells=4,formal_training_started=False,reward_changes=False,auxiliary_weight_changes=False))
    except Exception as error:
        pre.write_json(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--execute-preflight',action='store_true',required=True)
    args=p.parse_args();execute(args.parent_root.resolve(),args.output_root.resolve())
