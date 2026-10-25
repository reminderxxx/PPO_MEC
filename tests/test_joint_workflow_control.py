from copy import deepcopy
import pytest
from tests.test_ready_workflow import fixture
from src.envs.wrappers.joint_workflow_control import JointWorkflowEnv,JOINT_CONTRACT


def joint_fixture():
    env,state=fixture();env.__class__=JointWorkflowEnv
    return env,state


def test_service_and_future_preparation_overlap_with_real_cost():
    env,state=joint_fixture()
    state,r,_,_,info=env.step_joint('r0:a','adapter_fusion')
    assert 'r0:a' in state['workflow']['completed_node_ids']
    assert info['joint_control']['contract']==JOINT_CONTRACT
    assert info['persistent_stream']['transfer_mb']==64.
    assert any(x['object_id']=='adapter:fusion' and x['remaining_seconds']>0 for x in state['model_loading'])
    assert r.total==pytest.approx(sum(info['persistent_stream']['reward_terms'].values()))
    state,r,_,_,info=env.step_joint('r0:c',None)
    assert 'r0:c' not in state['workflow']['completed_node_ids']
    assert info['persistent_stream']['transfer_mb']==0.


def test_no_silent_current_admission_or_illegal_preparation():
    env,state=joint_fixture();env.step_joint('r0:a',None)
    with pytest.raises(ValueError,match='dependency-ready'):env.step_joint('r0:e',None)
    with pytest.raises(ValueError,match='Unreleased'):env.step_joint('r0:c','invented_adapter')
    state,_,_,_,info=env.step_joint('r0:c',None)
    assert 'r0:c' not in state['workflow']['completed_node_ids']
    assert info['persistent_stream']['transfer_mb']==0.


def test_default_step_node_remains_exactly_compatible():
    a,_=fixture();b,_=joint_fixture()
    for node in ['a','c','b','d','e','c','c','c']:
        x,r,d,t,i=a.step_node('r0:'+node);y,s,e,u,j=b.step_node('r0:'+node)
        assert x==y and r.total==s.total and d==e and t==u
        assert i==j
