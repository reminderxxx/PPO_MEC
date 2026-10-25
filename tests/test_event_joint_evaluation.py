from copy import deepcopy
import torch
import pytest
from scripts.evaluate_event_joint import JointRulePolicy
from tests.test_event_joint_learning import features
from src.envs.wrappers.joint_workflow_control import observe_joint_effects
from src.encoders.event_joint_encoder import event_features


def test_rules_choose_only_shared_explicit_candidates():
    env,s,_=features();f=event_features(observe_joint_effects(env,s),True)
    for name in ['no_prefetch','shortest_frontier','ready_demand','min_transfer','eviction_cost']:
        a=JointRulePolicy(name);i,_,_=a.act(f);assert f['mask'][i]
        assert i==(5 if name=='no_prefetch' else 2)
    with pytest.raises(ValueError):JointRulePolicy('unknown')


def test_no_candidate_cannot_trigger_a_hidden_fallback():
    env,s,_=features();f=event_features(observe_joint_effects(env,s),True);f['mask'][:5]=False
    for name in ['shortest_frontier','ready_demand','min_transfer','eviction_cost']:
        assert JointRulePolicy(name).act(f)[0]==5


def test_confirmation_guards_run_before_any_environment_interaction(tmp_path,monkeypatch):
    import argparse,json
    import scripts.evaluate_event_joint as evaluator
    from scripts.run_persistent_workflow_learning import sha
    inputs=tmp_path/'inputs.json';inputs.write_text(json.dumps(dict(units=[dict(index=2400,outer_window_index=24,case_index=0,split='confirmation',workflows=[])],workflows=[])))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(dict(input_sha256s=['wrong'],checkpoint_sha256=[],rules=['no_prefetch'])))
    counter=[]
    monkeypatch.setattr(evaluator,'episode',lambda *args,**kwargs:counter.append(1))
    args=argparse.Namespace(inputs=str(inputs),output=str(tmp_path/'reject_inputs'),split='confirmation',rule='no_prefetch',checkpoint=None,freeze=str(freeze))
    with pytest.raises(ValueError,match='input freeze'):evaluator.evaluate(args)
    freeze.write_text(json.dumps(dict(input_sha256s=[sha(inputs)],checkpoint_sha256=[],rules=['no_prefetch'])))
    args.output=str(tmp_path/'reject_rule');args.rule='min_transfer'
    with pytest.raises(ValueError,match='Unfrozen baseline'):evaluator.evaluate(args)
    assert counter==[]
    monkeypatch.setattr(evaluator,'episode',lambda *args,**kwargs:(dict(steps=0,reward=0.,completed_requests=0),[],[]))
    args.output=str(tmp_path/'matched');args.rule='no_prefetch';evaluator.evaluate(args)
    row=json.loads((tmp_path/'matched/rows.json').read_text())[0]
    assert row['window']==2400 and row['outer_window']==24 and row['case_index']==0


def test_support_matrix_contains_every_unseen_pair_without_outer_pseudoreplication():
    from scripts.prepare_event_joint_support import complete_pairs,build_payload
    pairs=complete_pairs();assert len(pairs)==len(set(pairs))==36
    assert len([p for p in pairs if p[0]==p[1]])==8
    payload=dict(units=[dict(index=24+i,split='confirmation',frames=[i]) for i in range(12)],all_workflows=[dict(workflow_id=f'w{i}') for i in range(16)])
    result=build_payload(payload,'hash')
    assert len({u['outer_window_index'] for u in result['units']})==12
    assert len({tuple(w['workflow_id'] for w in u['workflows']) for u in result['units']})==36
    assert all(int(w['workflow_id'][1:])>=8 for u in result['units'] for w in u['workflows'])


def test_final_candidate_freeze_cannot_open_confirmation_early(tmp_path):
    import json
    from scripts.finalize_event_joint_study import freeze_candidates,waiting_cells,ROOT
    plan_path=ROOT/'configs/experiment/event_joint_learning_v1.json';plan=json.loads(plan_path.read_text())
    assert len(waiting_cells(tmp_path,plan))==9
    with pytest.raises(ValueError,match='incomplete'):freeze_candidates(tmp_path,plan_path)
    assert not (tmp_path/'confirmation_freeze.json').exists()
