import numpy as np
import pytest
from scripts.analyze_event_joint_study import paired_hierarchical_bootstrap,sign_test,holm


def test_sign_test_ties_and_holm_global_family():
    assert sign_test([0.,1e-10,0.])['p_value']==1.
    assert sign_test(np.ones(12))['p_value']==pytest.approx(2/4096.)
    assert holm([.001,.02,.5])==pytest.approx([.003,.04,.5])


def test_constant_effect_and_zero_difference_bootstrap():
    for effect in [0.,3.]:
        x=paired_hierarchical_bootstrap(np.full((12,3),effect),499,7)
        assert x['estimate']==effect and x['percentile95']==[effect,effect] and x['bca95']==[effect,effect]
        assert x['outer_windows']==12 and x['inner_seeds']==3


def test_seed_repetition_does_not_inflate_outer_independent_units():
    d=np.arange(12.)[:,None]
    a=paired_hierarchical_bootstrap(d,999,7);b=paired_hierarchical_bootstrap(np.repeat(d,3,axis=1),999,7)
    assert a['outer_windows']==b['outer_windows']==12 and a['sign_test']==b['sign_test']
    assert a['percentile95']==b['percentile95'] and a['estimate']==b['estimate']


def test_pairing_retains_negative_effects_and_rejects_nonfinite():
    d=np.arange(-8.,4.)[:,None];r=paired_hierarchical_bootstrap(d,499,7)
    assert r['estimate']<0 and r['bca95'][0]<0
    with pytest.raises(ValueError):paired_hierarchical_bootstrap(np.array([[np.nan],[0.]]))


def test_workload_replication_is_nested_inside_time_windows():
    x=paired_hierarchical_bootstrap(np.ones((12,3,3)),499,7)
    assert x['outer_windows']==12 and x['inner_workloads']==3 and x['inner_seeds']==3
    assert x['sign_test']['p_value']==pytest.approx(2/4096.)
    assert x['bca95']==[1.,1.]


def test_complete_cohort_pairs_methods_workloads_and_seeds_without_duplication():
    import json
    from pathlib import Path
    from scripts.analyze_event_joint_study import cohort_statistics,ROOT
    plan=json.loads((ROOT/'configs/experiment/event_joint_learning_v1.json').read_text());plan['statistics']['bootstrap_replicates']=99
    records=[]
    for method in list(plan['methods'])+plan['causal_rules']:
        for case in range(36):
            for seed in (plan['seeds'] if method in plan['methods'] else [0]):
                primary=method==plan['primary_method']
                records.append(dict(method=method,seed=seed,window=2400+case,outer_window=24+case//3,reward=-15. if primary else -20.,completed_requests=6 if primary else 5,waiting_seconds=40. if primary else 50.,transfer_mb=400. if primary else 500.))
    table,comparisons,matrices=cohort_statistics(records,plan)
    assert len(comparisons)==28 and matrices['event_set_ppo']['reward'].shape==(12,3,3)
    assert table['event_set_ppo']['rows']==108 and table['shortest_frontier']['rows']==36
    reward=next(x for x in comparisons if x['metric']=='reward' and x['comparator']=='shortest_frontier')
    assert reward['estimate']==5. and reward['outer_windows']==12 and reward['bca95']==[5.,5.]


def test_end_to_end_analysis_keeps_nominal_and_effective_counts_separate(tmp_path,monkeypatch):
    import json
    import scripts.analyze_event_joint_study as module
    plan=json.loads((module.ROOT/'configs/experiment/event_joint_learning_v1.json').read_text());plan['statistics']['bootstrap_replicates']=99
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    for cohort,support in [('confirmation_v1',False),('supporting_confirmation_v1',True)]:
        rows=[]
        for method in list(plan['methods'])+plan['causal_rules']:
            for case in range(36 if support else 12):
                for seed in (plan['seeds'] if method in plan['methods'] else [0]):
                    good=method==plan['primary_method'];rows.append(dict(method=method,seed=seed,window=2400+case if support else 24+case,outer_window=24+case//3 if support else 24+case,reward=5. if good else 0.,completed_requests=6 if good else 5,waiting_seconds=40. if good else 50.,transfer_mb=400. if good else 500.))
        directory=tmp_path/cohort/'fixture';directory.mkdir(parents=True);(directory/'rows.json').write_text(json.dumps(rows))
    groups=[[24,28,32],[25,29,33],[26,30,34],[27,35],[31]]
    (tmp_path/'input_diversity_readonly_review.json').write_text(json.dumps(dict(input_effect_projection_classes=groups)))
    monkeypatch.setattr(module,'collect_learning',lambda *args:([dict(method='fixture',reward=0.)],[],[dict(native_steps=0)],{}))
    result=module.analyze(tmp_path,tmp_path/'report',plan_path)
    assert result['effective_primary_configurations']==5 and result['multiplicity_family_size']==56
    assert result['nominal_primary_comparisons'][0]['outer_windows']==12
    assert all(x['outer_windows']==5 and not x['positive_corrected_evidence'] for x in result['comparisons'])
    assert all(x['outer_windows']==12 and x['positive_corrected_evidence'] for x in result['supporting']['comparisons'])
