"""Read-only audit of the fixed auxiliary-supervision comparison."""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import torch
import run_fair_reward_training as fair
from preflight_sagh_execution_repair import write_json,CLASSES
from preflight_sagh_auxiliary_alignment import VARIANTS,SEEDS,TRAIN_CAP,VALIDATION_CAP

def records(p):return [json.loads(l) for l in p.open()]
def audit(root):
    e=json.loads((root/'execution_receipt.json').read_text());c=json.loads((root/'completion.json').read_text())
    assert e['train_cap']==TRAIN_CAP==3072 and e['validation_cap']==VALIDATION_CAP==2112
    assert not any(e[k] for k in ['reward_changes','auxiliary_weight_changes','old_checkpoint_loads','original_curve_changes','formal_training_started'])
    assert c['training_steps']==3072 and c['validation_steps']<=2112 and c['fresh_cells']==4
    assert c['total_environment_steps']==3072+c['validation_steps']
    source=Path(e['source_root'])
    for p,h in e['source_sha256'].items():assert fair.sha(source/p)==h,p
    result=[];initial={};configs={};prefixes={};observed_validation=0
    for variant in VARIANTS:
      for seed in SEEDS:
        cell=root/'variants'/variant/'cells/sa_ghmappo'/f'seed{seed}'
        complete=json.loads((cell/'complete.json').read_text())
        for p,h in complete['files_sha256'].items():assert fair.sha(cell/p)==h,p
        identity=json.loads((cell/'identity.json').read_text());cfg=dict(identity['config'])
        assert cfg.pop('auxiliary_action_contract')==variant
        assert cfg['execution_mapping_contract']=='shared_train_eval_v1'
        assert identity['fresh'] and identity['optimizer_initially_empty']
        configs[(variant,seed)]=cfg
        steps=records(cell/'steps.jsonl');phases=Counter();units=Counter();cl=defaultdict(set);episodes=defaultdict(lambda:[0,0.]);maxerr=0.;zero=Counter();fallback=Counter()
        for row in steps:
          m=row['metadata'];phase=m['phase'];phases[phase]+=1;cl[phase].add(m['window_class'])
          if phase=='train':units[m['unit']]+=1
          b=row['reward_breakdown'];ledger=sum(b[k] for k in ['positive_offset','service_reward','continuity_bonus','mechanism_exploration_bonus'])-sum(b[k] for k in ['delay_penalty','cache_miss_penalty','migration_cost','constraint_penalty'])
          maxerr=max(maxerr,abs(ledger-row['native_reward']));assert maxerr<1e-6 and b['positive_offset']==0
          key=(phase,m['unit'],m['training_steps']);episodes[key][0]+=1;episodes[key][1]+=row['native_reward']
          info=row['action_info'];assert info['auxiliary_action_contract']==variant
          assert 'idle_popularity_adapter_counts' in info['execution_mapping_state']
          assert row['action_mask'][row['actual_action']] and info['probability_contract']=='sampled_base_and_option_v1'
          lp=info['sampled_base_log_prob']+(info['option_gate']['option_log_prob'] if info['option_gate'].get('enabled') else 0)
          assert abs(lp-info['log_prob'])<2e-5
          zero[phase]+=row['actual_action']==0;fallback[phase]+=info['idle_popularity_fallback']['applied']
        for row in records(cell/'episodes.jsonl'):
          m=row['metadata'];n,total=episodes.pop((m['phase'],m['unit'],m['training_steps']))
          assert n==row['steps'] and abs(total-row['summary']['reward_breakdown']['total']['sum'])<1e-5
        assert not episodes and phases['train']==768 and phases['validation']==complete['validation_steps']
        assert sorted(units.values())==[128]*6 and all(cl[p]==set(CLASSES) for p in ['train','validation'])
        updates=records(cell/'updates.jsonl');assert [u['training_steps'] for u in updates]==[256,512,768]
        for u in updates:
          li=u['learn_info'];assert li['initial_log_prob_error_max']<2e-5 and li['accepted_policy_kl_max']<=cfg['target_kl']+1e-7
          dg=li['objective_diagnostics'];assert len(dg['credit_rows'])==256 and len(dg['gradient_batches'])==1
          assert dg['auxiliary_action_contract']==variant and len(dg['auxiliary_execution_last_seen_rows'])==256
          g=dg['gradient_batches'][0];assert math.isclose(g['weighted_term_sum'],g['total_loss'],rel_tol=1e-5,abs_tol=1e-5)
          assert all(math.isfinite(n) for t in g['terms'].values() for n in t['gradient_norm'].values())
          for row in dg['auxiliary_execution_last_seen_rows']:
            dd=row['diagnostics']
            if variant=='executed_marginal_v1':
              for objective in ['behavior','counterfactual']:
                if objective in dd:
                  v=dd[objective];assert math.isclose(sum(v['executed_probs']),1.,abs_tol=2e-5)
                  assert v['target_reachable']==(v['target_action'] in v['reachable_actions'])
                  if objective=='counterfactual':assert v['target_reachable']
        checkpoints=sorted(cell.glob('*.pt'));assert len(checkpoints)==4
        for cp in checkpoints:
          p=torch.load(cp,map_location='cpu',weights_only=False)
          assert p['config']['auxiliary_action_contract']==variant
          assert all(torch.isfinite(v).all() for v in p['network_state_dict'].values())
          assert p['update_count']==(0 if cp.stem=='initial' else int(cp.stem.split('_')[1]))
        first=torch.load(cell/'initial.pt',map_location='cpu',weights_only=False)
        last=torch.load(cell/'update_0003.pt',map_location='cpu',weights_only=False)
        assert not first['optimizer_state_dict']['state']
        initial[(variant,seed)]=first['network_state_dict']
        adam=[int(v['step']) for v in last['optimizer_state_dict']['state'].values() if 'step' in v]
        accepted=sum(u['learn_info']['accepted_optimizer_steps'] for u in updates)
        assert min(adam)>0 and max(adam)==accepted
        assert any(not torch.equal(v,last['network_state_dict'][k]) for k,v in first['network_state_dict'].items())
        probes=records(cell/'marginal_probes.jsonl');assert len(probes)==6
        assert all(math.isclose(sum(p['diagnostic']['executed_action_probs']),1,abs_tol=2e-5) for p in probes)
        # The arms must have identical exposure/behavior before the first update.
        prefixes[(variant,seed)]=[{k:r[k] for k in ['metadata','semantic_state','actual_action','native_reward','reward_breakdown']} for r in steps if r['metadata']['phase']=='train' and r['metadata']['unit']<2]
        observed_validation+=phases['validation']
        result.append(dict(variant=variant,seed=seed,train_steps=768,validation_steps=phases['validation'],
            zero_actions=dict(zero),fallback=dict(fallback),checkpoints=4,learner_opportunities=3,
            accepted_optimizer_steps=accepted,native_ledger_max_error=maxerr))
    for seed in SEEDS:
      a,b=VARIANTS
      assert configs[(a,seed)]==configs[(b,seed)]
      assert prefixes[(a,seed)]==prefixes[(b,seed)]
      assert all(torch.equal(v,initial[(b,seed)][k]) for k,v in initial[(a,seed)].items())
    assert observed_validation==c['validation_steps']
    return dict(status='PASS',reviewed_at=datetime.now().astimezone().isoformat(),source_commit=e['source_commit'],cells=result,
        initial_weights_equal_by_seed=True,identical_first_256_train_steps=True,only_contract_differs=True,
        formal_training_started=False,reward_and_auxiliary_weights_unchanged=True,performance_and_convergence='UNVERIFIED')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args();report=audit(a.run_root);write_json(a.run_root/'preflight_audit.json',report);print('PASS 4 cells / 16 checkpoints / 3072 train steps')
