"""Pure software fixtures: no torch/project import, env, scoring, or optimizer."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('offline_credit',Path(__file__).resolve().parents[1]/'scripts/audit_sa_stage5_credit.py')
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def fixture(first=False, forced=False):
    ids=[1] if forced else [1,3]; mask=[False,True,False,not forced]
    phases=M.PHASES[0 if first else 1]
    actions=[[1]] if first else [[1,1,1]]
    variants={}
    for label in M.LABELS:
        pv=[1.] if forced else ([.5,.5] if label=='reference_prior_only' else [.25,.75])
        valid_logits=[0.] if forced else ([0.,0.] if label=='reference_prior_only' else [math.log(.25),math.log(.75)])
        probs=[0.,pv[0],0.,0. if forced else pv[1]]
        logs=[-1e9,math.log(pv[0]),-1e9,-1e9 if forced else math.log(pv[1])]
        entropy=-sum(x*math.log(x) for x in pv if x)
        q=[.1] if forced else [.1,.9]; sur=[[.3,2.,3.]] if forced else [[.3,2.,3.],[.7,4.,5.]]
        baseline=sum(p*.5*(v+s[0]) for p,v,s in zip(pv,q,sur))
        winner=1 if forced or label=='reference_prior_only' else 3
        heads=[]
        for j in range(1 if first else 3):
            heads.append(dict(valid_candidate_ids=ids,valid_candidate_count=len(ids),actual_mask=mask,
                valid_total_logits=valid_logits,valid_learned_logits=valid_logits if label!='reference_prior_only' else [0.]*len(ids),
                valid_public_prior=[0.]*len(ids),valid_other_adjustment=[0.]*len(ids),
                greedy_action=winner,reference_action=1,reference_action_probability=pv[0],
                public_prior_argmax=1,learned_only_argmax=winner,overrides_prior=winner!=1,
                prior_changes_learned_winner=False,top2_margin=None if forced else (0. if label=='reference_prior_only' else math.log(3)),
                distribution=dict(probabilities=[probs],log_probabilities=[logs],entropy=[entropy],
                    normalized_entropy=[entropy/max(math.log(len(ids)),1)],reference_log_probability=[logs[1]]),
                valid_candidate_surrogate_estimates=sur,valid_candidate_q=None if first else [[v] for v in q],
                prefix_value=None if first else [[.2]]))
        output=dict(reference_head_log_probs=[[logs[1]]*(1 if first else 3)],critic_value=None if first else [[.4]])
        if not first:
            output.update(normalized_entropies=[[entropy/max(math.log(len(ids)),1)]*3],value=[.4],
                selected_surrogate=[[sur[0]]*3],prefix_values=[[.2]*3],active_heads=[[not forced]*3],
                counterfactual_baselines=[[baseline]*3],selected_q=[[q[0]]*3])
        variants[label]=dict(heads=heads,head_count=len(heads),diagnostic_forward_count=1,
            guards=dict(batch_unchanged=True,weights_unchanged=True,rng_unchanged=True),forward_outputs=output)
    return dict(schema=M.SCHEMA+'_scores',witness=dict(arm='sa_legacy_clip',agent='sa_ghmappo',rule='test',seed=17,episode=64,case_id='dev_046'),
        frame=220,phase=phases,vehicle_id='v1',node_id='node',state_identity={},reference_model='sa_legacy_clip:64',
        reference_actions=actions,candidates=[['x','r1','x','r2'],['x','c1','x','c2'],[[None,0],[None,1],[None,2],[None,3]]],
        variants=variants,reference_parity=True,initial_arm_parity=True,
        guards=[dict(label=k,state_before='a',state_after='a') for k in M.LABELS],
        episode_labels={} if first else dict(returns=[8.],team_credit_targets=[1.],credit_targets=[1.]))


def team(controls,frame):
    return dict(ordered_team_controls=[dict(vehicle_id=v,control=c) for v,c in controls.items()],
                physical_priority_order=M.physical_order(controls,frame))


class CreditTests(unittest.TestCase):
    def test_second_pass_maps_sparse_ids_and_baseline(self):
        rows=M.analyze_score(fixture())
        self.assertEqual(len(rows),21)
        ref=next(x for x in rows if x['model']=='sa_legacy_clip:64' and x['head']==1)
        self.assertAlmostEqual(ref['counterfactual_baseline'],.65)
        self.assertEqual(ref['reference_action_conditional_rank'],2)
        self.assertAlmostEqual(ref['factual_reference_calibration']['signed_errors']['selected_q'],-.9)
        self.assertEqual(ref['candidates'][1]['candidate_id'],3)
        self.assertEqual(ref['candidates'][1]['ranks']['q'],1)

    def test_other_model_and_prior_only_no_factual_error(self):
        rows=M.analyze_score(fixture())
        self.assertEqual(sum(x['factual_reference_calibration'] is not None for x in rows),3)
        for x in rows:
            if x['model']!='sa_legacy_clip:64': self.assertIsNone(x['factual_reference_calibration'])

    def test_legacy_qualification_witness_uses_explicit_reference_model(self):
        row=fixture(); del row['witness']['arm']; del row['witness']['episode']
        row['witness']['method']='sa_ghmappo'
        out=M.analyze_score(row)
        self.assertTrue(all(x['reference_arm']=='sa_legacy_clip' and x['reference_episode']==64 for x in out))
        row['witness']['arm']='sa_group_clip'
        with self.assertRaises(ValueError): M.analyze_score(row)

    def test_firstpass_keeps_surrogate_but_q_and_v_unavailable(self):
        rows=M.analyze_score(fixture(first=True)); self.assertEqual(len(rows),7)
        for x in rows:
            self.assertIsNone(x['selected_q']); self.assertIsNone(x['global_value'])
            self.assertIsNone(x['factual_reference_calibration']); self.assertIsNotNone(x['candidates'][0]['surrogate_return'])

    def test_fabricated_first_q_rejected(self):
        row=fixture(first=True); row['variants']['sa_legacy_clip:64']['heads'][0]['valid_candidate_q']=[[1],[2]]
        with self.assertRaises(ValueError): M.analyze_score(row)

    def test_forced_active_and_entropy(self):
        rows=M.analyze_score(fixture(forced=True))
        self.assertTrue(all(x['forced'] and x['conditional_entropy']==0 for x in rows))
        row=fixture(forced=True); row['variants']['sa_legacy_clip:64']['forward_outputs']['active_heads'][0][2]=True
        with self.assertRaises(ValueError): M.analyze_score(row)

    def test_tampered_baseline_and_candidate_mapping_rejected(self):
        row=fixture(); row['variants']['sa_legacy_clip:64']['forward_outputs']['counterfactual_baselines'][0][1]+=1
        with self.assertRaises(ValueError): M.analyze_score(row)
        row=fixture(); row['variants']['sa_legacy_clip:64']['heads'][0]['valid_candidate_ids']=[0,1]
        with self.assertRaises(ValueError): M.analyze_score(row)

    def test_mc10_not_misused_as_value_target(self):
        ref=next(x for x in M.analyze_score(fixture()) if x['model']=='sa_legacy_clip:64')
        self.assertEqual(ref['factual_reference_calibration']['target_team_mc_over_80'],1)
        self.assertEqual(ref['factual_reference_calibration']['factual_team_mc_over_10'],8)

    def test_float32_reduction_tolerance_not_effect_tolerance(self):
        M.close(.65000004,.65)
        with self.assertRaises(ValueError): M.close(.66,.65)

    def test_large_common_logit_offset_roundoff_bound(self):
        logp=-0.05829975757976529; saved=-0.05829620361328125
        with self.assertRaises(ValueError): M.close(logp,saved)
        M.close(logp,saved,atol=M.categorical_roundoff_atol([100.,97.]))
        with self.assertRaises(ValueError): M.close(logp,saved+.01,atol=M.categorical_roundoff_atol([100.,97.]))

    def test_float32_subnormal_probability_has_absolute_ulp_bound(self):
        logp=-99.61536825730221;saved=-99.61902618408203;p=math.exp(logp)
        with self.assertRaises(ValueError):M.close(logp,saved,atol=M.categorical_roundoff_atol([-99.62,0.]))
        bound=M.log_probability_roundoff_atol([-99.62,0.],p)
        M.close(logp,saved,atol=bound)
        with self.assertRaises(ValueError):M.close(logp,saved+.1,atol=bound)

    def test_rank_ties_use_lowest_valid_id(self):
        self.assertEqual(M.ranking([1,3,4],[1.,1.,.2]),[1,3,4])

    def test_single_owner_four_priorities_class_and_prepare_separate(self):
        controls={'v1':dict(target='x',prepare_target=None,priority=1,cache_target='c')}
        candidates=[[r,p] for r in (None,'r1') for p in range(4)]
        before=deepcopy(controls)
        e=M.priority_classes(candidates,list(range(8)),'v1',team(controls,220),220)
        self.assertEqual(e['sufficient_class_count'],5); self.assertEqual(e['class_redundancy_lower_bound'],3)
        self.assertEqual(e['classes'][0]['candidate_ids'],[0,1,2,3]); self.assertTrue(e['single_active_control'])
        self.assertTrue(all(g['equivalence_status']=='nonreference_prepare_UNVERIFIED_singleton' for g in e['classes'][1:]))
        self.assertEqual(controls,before)

    def test_rotating_tie_and_order_change_not_merged(self):
        controls={'v1':dict(target='x',prepare_target=None,priority=1),'v2':dict(target='y',prepare_target=None,priority=1)}
        # Oddframe tie order is v2,v1: owner p0/p1 same order; p2/p3 flip.
        e=M.priority_classes([[None,p] for p in range(4)],list(range(4)),'v1',team(controls,221),221)
        self.assertEqual([g['candidate_ids'] for g in e['classes']],[[0,1],[2,3]])
        self.assertFalse(e['single_active_control']); self.assertEqual(e['reference_global_order'],['v2','v1'])

    def test_recorded_global_order_tamper_rejected(self):
        controls={'v1':dict(prepare_target=None,priority=1),'v2':dict(prepare_target=None,priority=1)}
        t=team(controls,221); t['physical_priority_order']=['v1','v2']
        with self.assertRaises(ValueError): M.priority_classes([[None,0]], [0], 'v1', t,221)

    def test_aggregate_keeps_negative_seed_and_case(self):
        agg=M.Aggregate()
        for row in M.analyze_score(fixture()): agg.add(row)
        out=agg.finish(); self.assertEqual(len(out),21)
        self.assertTrue(all(x['seed']==17 and x['case_id']=='dev_046' for x in out))
        self.assertEqual(sum(x['factual_reference_calibration'] is not None for x in out),3)

    def test_complete_gate_precedes_tensor_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'contract.json').write_text('{"status":"frozen"}')
            (p/'result.json').write_text('{"status":"running"}')
            with self.assertRaises(ValueError): M.verify_complete(p)

    def test_pure_hash_preserves_types_and_order(self):
        self.assertEqual(M.object_hash({'a':[1],'b':(2,)}),M.object_hash({'b':(2,),'a':[1]}))
        self.assertNotEqual(M.object_hash([1]),M.object_hash((1,)))


if __name__=='__main__': unittest.main()
