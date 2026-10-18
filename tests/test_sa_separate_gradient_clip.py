"""Stage3 software acceptance only: synthetic tensors, no scientific training.

Requires the separately authorized candidate installation. This file itself is
not an implementation patch. Adam.step has a module-session ceiling of128.
Optional SA_STAGE3_TEST_RECEIPT writes a new /tmp JSON receipt at teardown.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agents.ppo_agent import StreamControlPPO
from src.agents.sa_ghmappo_agent import StreamSAGHMAPPOAgent
from src.agents.registry import build_agent

VALUE_PREFIXES = ('critic.', 'prefix_values.', 'action_values.')
TAG = 'sa_policy_value_clip_v1'


@pytest.fixture(scope='module', autouse=True)
def software_budget_guard(request):
    """Real Adam calls are counted even when reached through inherited methods."""
    from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
    from src.envs.core.multi_request_replay_env import MultiRequestReplayEnv
    counts = dict(adam_steps=0, adam_step_limit=128, environment_calls=0,
                  scientific_training_sessions=0, synthetic_update_calls=0)
    patch = pytest.MonkeyPatch()
    original_step = torch.optim.Adam.step
    original_update = StreamControlPPO.update
    original_threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    receipt = os.environ.get('SA_STAGE3_TEST_RECEIPT')
    if receipt:
        receipt = Path(receipt).resolve()
        if not receipt.is_relative_to(Path('/tmp').resolve()):
            raise ValueError('SA_STAGE3_TEST_RECEIPT must be under /tmp')
        if receipt.exists():
            raise FileExistsError(receipt)

    def counted_step(optimizer, *args, **kwargs):
        if counts['adam_steps'] >= counts['adam_step_limit']:
            raise RuntimeError('stage3 software Adam.step budget exhausted')
        counts['adam_steps'] += 1
        return original_step(optimizer, *args, **kwargs)

    def counted_update(agent, *args, **kwargs):
        counts['synthetic_update_calls'] += 1
        return original_update(agent, *args, **kwargs)

    def forbidden_environment(*args, **kwargs):
        counts['environment_calls'] += 1
        raise RuntimeError('stage3 software tests prohibit environment construction/reset/step')

    patch.setattr(torch.optim.Adam, 'step', counted_step)
    patch.setattr(StreamControlPPO, 'update', counted_update)
    patch.setattr(StreamControlPPO, 'act_controls', forbidden_environment)
    for cls in (MultiRequestStreamEnv, MultiRequestReplayEnv):
        for name in ('__init__', 'reset', 'step', 'step_controls'):
            if hasattr(cls, name):
                patch.setattr(cls, name, forbidden_environment)
    torch.set_num_threads(1)
    try:
        yield counts
    finally:
        patch.undo()
        torch.set_num_threads(original_threads)
        torch.set_rng_state(rng)
        counts.update(schema='sa_stage3_software_test_receipt_v1',
                      module=str(Path(__file__).resolve()),
                      pytest_session_failures_observed=int(request.session.testsfailed),
                      budget_respected=counts['adam_steps'] <=128,
                      environment_steps=0,
                      synthetic_only=True)
        if receipt:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            with receipt.open('x') as f:
                json.dump(counts, f, indent=2); f.write('\n')


def agent(enabled=False, **kwargs):
    return StreamSAGHMAPPOAgent(seed=17, training_budget=64, return_gamma=1.,
                               separate_policy_value_clip=enabled, **kwargs)


def synthetic_data(batch=6):
    g = torch.Generator().manual_seed(521)
    d = dict(own=torch.rand(batch,20,generator=g)*.3,
             critic=torch.rand(batch,36,generator=g)*.3,
             graph=torch.rand(batch,64,4,generator=g)*.3,
             adjacency=torch.eye(64).repeat(batch,1,1),
             graph_mask=torch.arange(64).repeat(batch,1)<6,
             domain=torch.rand(batch,4,generator=g)*.3,
             prior0=torch.tensor([0.,-.5,-1.,-.7,-.2,0.]).repeat(batch,1),
             opportunity_count=torch.full((batch,),2))
    for j,n in enumerate((6,9,20)):
        f=torch.rand(batch,n,32,generator=g)*.3
        d[f'f{j}']=f; d[f'm{j}']=torch.ones(batch,n,dtype=torch.bool)
        d[f'node{j}']=torch.arange(n).remainder(6).repeat(batch,1)
        f[...,0]=0; f[...,16:20]=0; f[...,12]=1; f[...,14]=0
        for k in range(n):
            site=(k-1)%4 if j!=2 else k//4-1
            if ((j!=2 and k>0 and k!=5) or (j==2 and site>=0)):
                f[:,k,0]=1; f[:,k,16+site]=1
        f[...,4]=1; f[...,6]=1; f[...,5]=0
    d['m0'][:,[0,5]]=False
    d['m0'][::3]=False; d['m0'][::3,1]=True
    d['f0'][1::3,:,4]=0; d['f0'][1::3,:,6]=0; d['own'][:,9]=0
    d['f2'][:,:4,0]=0; d['f2'][:,:4,16:20]=0
    return d


def synthetic_rollouts(model):
    data=synthetic_data()
    with torch.no_grad():
        a,logs,ent,v,sur,prefix,active,baseline,q,_=model._evaluate(data,deterministic=True,return_details=True)
    target=torch.linspace(.5,4.,len(a))
    return [dict(data=data,actions=a,head_logs=logs,values=v,active_heads=active,
                 counterfactual_baselines=baseline,returns=target*8,
                 credit_targets=target.clone(),team_credit_targets=target.clone(),
                 cost_targets=torch.zeros(len(a),2),initial_request_count=8,
                 episode_end=True)]


def assert_exact(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for left,right in zip(a,b): assert_exact(left,right)
    else:
        assert a==b


def grad_snapshot(model):
    return {n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}


def test_default_matches_parent_including_exact_synthetic_update():
    legacy=StreamControlPPO(method='sa_ghmappo',seed=17,training_budget=64,return_gamma=1.)
    legacy_rng=torch.get_rng_state().clone()
    default=StreamSAGHMAPPOAgent(seed=17,training_budget=64,return_gamma=1.)
    assert torch.equal(legacy_rng,torch.get_rng_state())
    explicit=agent(False)
    assert_exact(legacy.state_dict(),default.state_dict()); assert_exact(default.state_dict(),explicit.state_dict())
    assert_exact(legacy.options,default.options); assert_exact(legacy.options,explicit.options)
    assert 'separate_policy_value_clip' not in default.options
    rows=synthetic_rollouts(legacy)
    torch.manual_seed(188); before=torch.get_rng_state().clone()
    result_legacy=legacy.update(deepcopy(rows),epochs=1)
    after=torch.get_rng_state().clone()
    torch.set_rng_state(before); result_default=default.update(deepcopy(rows),epochs=1)
    assert torch.equal(after,torch.get_rng_state())
    assert_exact(result_legacy,result_default)
    assert_exact(legacy.state_dict(),default.state_dict())
    assert_exact(legacy.optimizer.state_dict(),default.optimizer.state_dict())
    assert_exact(legacy.surrogate_optimizer.state_dict(),default.surrogate_optimizer.state_dict())
    assert_exact(grad_snapshot(legacy),grad_snapshot(default))


def test_true_forward_same_and_groups_partition_original_adam():
    old=agent(False); new=agent(True)
    assert_exact(old.state_dict(),new.state_dict())
    with torch.no_grad():
        left=old._evaluate(synthetic_data(),deterministic=True,return_details=True)
        right=new._evaluate(synthetic_data(),actions=left[0],deterministic=True,return_details=True)
    assert_exact(left,right)
    policy={id(p) for p in new._separate_policy_parameters}
    value={id(p) for p in new._separate_value_parameters}
    surrogate={id(p) for p in new.surrogates.parameters()}
    assert (len(policy),len(value),len(surrogate))==(21,28,12)
    assert not policy&value and not policy&surrogate and not value&surrogate
    assert policy|value|surrogate=={id(p) for p in new.parameters()}
    assert [id(p) for g in new.optimizer.param_groups for p in g['params']]==[id(p) for n,p in new.named_parameters() if not n.startswith('surrogates.')]
    assert len(new.optimizer.param_groups)==1
    assert_exact(old.optimizer.state_dict(),new.optimizer.state_dict())
    expected_value={id(p) for n,p in new.named_parameters() if n.startswith(VALUE_PREFIXES)}
    assert value==expected_value


def test_loss_gradient_routes_have_no_shared_trainable_parameters():
    model=agent(True); out=model._evaluate(synthetic_data(),deterministic=True,return_details=True)
    _,logs,ent,value,sur,prefix,active,_,q,_=out
    losses={'policy_entropy':-logs[active].mean()-.01*ent[active].mean(),
            'value':value.square().mean()+prefix.square().mean()+q.square().mean(),
            'surrogate':sur.square().mean()}
    named=list(model.named_parameters())
    for kind,loss in losses.items():
        grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True,retain_graph=True)
        positive=0
        for (name,p),g in zip(named,grads):
            group='surrogate' if name.startswith('surrogates.') else ('value' if name.startswith(VALUE_PREFIXES) else 'policy_entropy')
            if g is not None:
                assert torch.isfinite(g).all()
                if group!=kind: assert torch.count_nonzero(g)==0
                else: positive+=int(torch.count_nonzero(g))
        assert positive>0


def mocked_gradient_application(monkeypatch, enabled, value_scale):
    model=agent(enabled); captured={}; calls=[]
    def main_step(*args,**kwargs):
        calls.append('main'); captured.update(grad_snapshot(model))
    def surrogate_step(*args,**kwargs): calls.append('surrogate')
    monkeypatch.setattr(model.optimizer,'step',main_step)
    monkeypatch.setattr(model.surrogate_optimizer,'step',surrogate_step)
    named=dict(model.named_parameters())
    names=('heads.0.0.weight','critic.0.weight','surrogates.0.0.weight')
    before=deepcopy(model.state_dict())
    loss=.3*named[names[0]].flatten()[0]+value_scale*named[names[1]].flatten()[0]+.2*named[names[2]].flatten()[0]
    model._apply_control_gradients(loss)
    assert calls==['main','surrogate']; assert_exact(before,model.state_dict())
    assert model.optimizer.state_dict()['state']=={}
    return model, [float(captured[n].flatten()[0]) for n in names]


def test_independent_pre_step_clipping_and_legacy_value_coupling(monkeypatch):
    small,small_grad=mocked_gradient_application(monkeypatch,True,2.)
    large,large_grad=mocked_gradient_application(monkeypatch,True,200.)
    _,legacy_small=mocked_gradient_application(monkeypatch,False,2.)
    _,legacy_large=mocked_gradient_application(monkeypatch,False,200.)
    assert small_grad[0]==large_grad[0]==pytest.approx(.3)
    assert small_grad[2]==large_grad[2]==pytest.approx(.2)
    for model,grads,norm in [(small,small_grad,2.),(large,large_grad,200.)]:
        assert abs(grads[1])<=.5
        assert grads[1]==pytest.approx(norm*.5/(norm+1e-6),abs=2e-7)
        stats=model._separate_gradient_stats[-1]
        assert stats[3]==1 and stats[5]==1
        assert stats[4]==pytest.approx(.5/(norm+1e-6))
    assert legacy_large[0]<legacy_small[0]/50
    _,under=mocked_gradient_application(monkeypatch,True,.1)
    assert under==pytest.approx([.3,.1,.2])


@pytest.mark.parametrize('enabled',[False,True])
def test_checkpoint_nonempty_adam_roundtrip_and_legacy_metadata(tmp_path,enabled):
    model=agent(enabled)
    model.update(synthetic_rollouts(model),epochs=1)
    assert model.optimizer.state_dict()['state'] and model.surrogate_optimizer.state_dict()['state']
    path=tmp_path/'checkpoint.pt'; model.save_checkpoint(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['training_resume_supported'] is False
    if enabled:
        assert saved['options']['separate_policy_value_clip'] is True
        assert saved['training_gradient_contract']==TAG
    else:
        assert 'separate_policy_value_clip' not in saved['options']
        assert 'training_gradient_contract' not in saved
    restored=type(model).from_checkpoint(path)
    assert restored.separate_policy_value_clip is enabled
    assert_exact(model.state_dict(),restored.state_dict())
    assert_exact(model.optimizer.state_dict(),restored.optimizer.state_dict())
    assert_exact(model.surrogate_optimizer.state_dict(),restored.surrogate_optimizer.state_dict())
    restored.save_checkpoint(tmp_path/'resaved.pt')
    assert_exact(saved,torch.load(tmp_path/'resaved.pt',map_location='cpu',weights_only=False))


@pytest.mark.parametrize('mode',['true_missing','true_unknown','false_tag','absent_flag_tag','string_flag','int_flag'])
def test_checkpoint_flag_tag_negative_cases(tmp_path,mode):
    model=agent(True); path=tmp_path/'bad.pt'; model.save_checkpoint(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if mode=='true_missing': saved.pop('training_gradient_contract')
    elif mode=='true_unknown': saved['training_gradient_contract']='unknown'
    elif mode=='false_tag': saved['options']['separate_policy_value_clip']=False
    elif mode=='absent_flag_tag': saved['options'].pop('separate_policy_value_clip')
    elif mode=='string_flag': saved['options']['separate_policy_value_clip']='false'
    elif mode=='int_flag': saved['options']['separate_policy_value_clip']=1
    torch.save(saved,path)
    with pytest.raises(ValueError): StreamSAGHMAPPOAgent.from_checkpoint(path)


def test_registry_numeric_metrics_and_other_algorithms_unchanged():
    model=build_agent('sa_ghmappo',control_contract='joint_stream_control_v2',seed=17,
                      training_budget=64,return_gamma=1.,separate_policy_value_clip=True)
    assert type(model) is StreamSAGHMAPPOAgent and model.separate_policy_value_clip
    result=model.update(synthetic_rollouts(model),epochs=1)
    assert result['gradient_steps']==1 and result['separate_policy_value_clip']==1
    for value in result.values():
        assert isinstance(value,(int,float)) and math.isfinite(value)
    for name in ('policy_gradient_norm','value_gradient_norm','policy_clip_coefficient','value_clip_coefficient'):
        assert name in result and result[name]>0
    for name in ('mappo','ppo','cache_offload_drl','dag_offload_drl','dt_handoff_drl','dqn'):
        other=build_agent(name,control_contract='joint_stream_control_v2',seed=17,training_budget=64,return_gamma=1.)
        assert 'separate_policy_value_clip' not in other.options


def test_kl_stop_metrics_are_numeric_without_optimizer_step(software_budget_guard):
    model=agent(True); rows=synthetic_rollouts(model)
    rows[0]['head_logs']=rows[0]['head_logs']-10
    before=software_budget_guard['adam_steps']
    result=model.update(rows,epochs=1)
    assert result['kl_early_stop']==1 and result['gradient_steps']==0
    assert software_budget_guard['adam_steps']==before
    assert result['policy_gradient_norm']==0 and result['value_gradient_norm']==0
    assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in result.values())


def test_disabled_surrogate_skips_its_optimizer(software_budget_guard):
    model=agent(True,surrogate_enabled=False)
    before=software_budget_guard['adam_steps']
    result=model.update(synthetic_rollouts(model),epochs=1)
    assert result['gradient_steps']==1
    assert software_budget_guard['adam_steps']-before==1
    assert model.surrogate_optimizer.state_dict()['state']=={}


@pytest.mark.parametrize('value',['false',0,1])
def test_constructor_rejects_nonboolean_flag(value):
    with pytest.raises(ValueError,match='must be a bool'):
        agent(value)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('connected_zero', [False, True])
def test_all_none_and_zero_gradients_are_stable(enabled, connected_zero):
    model = agent(enabled)
    before = deepcopy(model.state_dict())
    loss = (next(model.parameters()).sum()*0 if connected_zero
            else torch.tensor(0., requires_grad=True))
    model._apply_control_gradients(loss)
    assert_exact(before, model.state_dict())
    assert all(math.isfinite(v) for row in model._gradient_stats for v in row)
    if enabled:
        assert model._separate_gradient_stats[-1][:3] == (0., 0., 0.)


def test_no_prediction_default_retains_parent_update():
    from scripts.train_joint_stream_control import make_agent
    default = make_agent('sa_no_prediction', 17, 64)
    parent = StreamControlPPO(**default.options)
    assert not default.prediction_enabled and not default.separate_policy_value_clip
    assert_exact(parent.state_dict(), default.state_dict())
    rows = synthetic_rollouts(parent)
    rng = torch.get_rng_state().clone()
    expected = parent.update(deepcopy(rows), epochs=1)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    got = default.update(deepcopy(rows), epochs=1)
    assert torch.equal(after, torch.get_rng_state())
    assert_exact(expected, got)
    assert_exact(parent.state_dict(), default.state_dict())
    assert_exact(parent.optimizer.state_dict(), default.optimizer.state_dict())
