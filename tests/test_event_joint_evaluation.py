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
