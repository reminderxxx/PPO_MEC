"""Pure software Stage5B contract/guard checks. No models, tensors or environments."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/diagnose_sa_stage5_same_state.py'
spec=importlib.util.spec_from_file_location('stage5_diag_under_test',SCRIPT)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def fixture_plan():
    sessions=[]
    for seed in d.SEEDS:
        for arm in d.ARMS:
            for ep in d.EPISODES:
                for cid in d.CASES:
                    row=dict(steps=200,reward=1.0)
                    identity=dict(arm=arm,agent='sa_ghmappo',rule=f'{arm}_seed{seed}',seed=seed,episode=ep,case_id=cid)
                    sessions.append(dict(arm=arm,seed=seed,episode=ep,case_id=cid,identity=identity,
                        frames=d.quantile_frames(200),expected_summary=row,expected_summary_sha256=d.digest(row),
                        expected_ledger=dict(receipt_frames=200)))
    cp={str(seed):{d.model_key(arm,ep):dict(completed_updates=ep//4,expected_updates=ep//4,options_flag=flag)
                  for arm,flag in d.ARMS.items() for ep in d.EPISODES} for seed in d.SEEDS}
    return dict(schema=d.SCHEMA,budgets=deepcopy(d.CAPS),arms=deepcopy(d.ARMS),sessions=sessions,checkpoints=cp,
                scoring_conditions=[d.model_key(a,e) for a in d.ARMS for e in d.EPISODES]+['reference_prior_only'])


def test_frozen_cartesian_and_actual_length_quantiles():
    p=fixture_plan();d.verify_shape(p)
    assert len(p['sessions'])==144
    assert d.quantile_frames(480)==[0,59,119,179,239,299,359,419,479]
    assert d.quantile_frames(200)==[0,24,49,74,99,124,149,174,199]
    assert d.quantile_frames(1)==[0]
    assert d.quantile_frames(3)==[0,1,2]
    for n in (0,481,True,2.5):
        with pytest.raises(ValueError):d.quantile_frames(n)


@pytest.mark.parametrize('kind',['extra_session','duplicate_session','case','frame','row','row_hash','trace_frames','budget','arms','scorers','cp_flag','cp_updates'])
def test_reject_changed_frozen_scope(kind):
    p=fixture_plan()
    if kind=='extra_session':p['sessions'].append(deepcopy(p['sessions'][0]))
    elif kind=='duplicate_session':p['sessions'][1]=deepcopy(p['sessions'][0])
    elif kind=='case':p['sessions'][0]['case_id']='dev_047'
    elif kind=='frame':p['sessions'][0]['frames'][-1]=479
    elif kind=='row':p['sessions'][0]['expected_summary']['reward']=2
    elif kind=='row_hash':p['sessions'][0]['expected_summary_sha256']='bad'
    elif kind=='trace_frames':p['sessions'][0]['expected_ledger']['receipt_frames']=201
    elif kind=='budget':p['budgets']['reference_frames']+=1
    elif kind=='arms':p['arms']['sa_group_clip']=False
    elif kind=='scorers':p['scoring_conditions'].pop()
    elif kind=='cp_flag':p['checkpoints']['7']['sa_legacy_clip:0']['options_flag']=True
    elif kind=='cp_updates':p['checkpoints']['7']['sa_group_clip:32']['completed_updates']=32
    with pytest.raises(ValueError):d.verify_shape(p)


def test_charge_before_operation_no_refund():
    b=d.Budget({'frames':1});b.charge('frames')
    with pytest.raises(RuntimeError):b.charge('frames')
    assert b.used=={'frames':1}
    with pytest.raises(RuntimeError):b.charge('frames',-1)


def test_sha_changed_and_unicode_canonical_matches_original_auditor(tmp_path):
    p=tmp_path/'x';p.write_text('original');binding=d.bound(p);d.check_bound(binding)
    p.write_text('changed')
    with pytest.raises(ValueError):d.check_bound(binding)
    row={'arm':'ignored','seed':7,'case_id':'ignored','label':'中文','nested':{'seed':9}}
    cleaned={'label':'中文','nested':{'seed':9}}
    expected=(json.dumps(cleaned,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    assert d.canonical(d.clean_payload(row))==expected
    assert b'\\u' in expected
    gz=tmp_path/'events.gz'
    with gzip.open(gz,'wt') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    import hashlib
    assert d.trace_digest(gz)=={'sha256':hashlib.sha256(expected).hexdigest(),'rows':1}


def test_priority_same_prepare_can_change_physical_team_order():
    controls={v:SimpleNamespace(priority=p,prepare_target=None) for v,p in [('v0',0),('v1',0),('v2',0)]}
    assert d.physical_priority_order(controls,1)==['v1','v2','v0']
    controls['v0'].priority=3
    assert d.physical_priority_order(controls,1)==['v0','v1','v2']
    assert d.physical_priority_order({},1)==[]


@contextmanager
def fake_patch(obj,name,fn):
    had=name in obj.__dict__;old=obj.__dict__.get(name)
    setattr(obj,name,MethodType(fn,obj))
    try:yield
    finally:
        if had:setattr(obj,name,old)
        else:delattr(obj,name)


@pytest.mark.parametrize('exception',[False,True])
def test_all_learning_guards_and_restoration(exception):
    def okay(*a,**k):return 'okay'
    class Tensor:backward=okay
    class Base:step=okay
    class Adam(Base):step=okay
    class AdamW(Base):step=okay
    torch=SimpleNamespace(Tensor=Tensor,autograd=SimpleNamespace(backward=okay,grad=okay),
        optim=SimpleNamespace(Optimizer=Base,Adam=Adam,AdamW=AdamW),no_grad=nullcontext)
    agent=SimpleNamespace(update=okay,optimizer=Adam(),surrogate_optimizer=Adam())
    rt={'torch':torch,'score':SimpleNamespace(patched_method=fake_patch)}
    before=(Tensor.backward,Base.step,Adam.step,AdamW.step,torch.autograd.backward,torch.autograd.grad,agent.update)
    class Deliberate(Exception):pass
    try:
        with d.no_learning(rt,{'m':agent}):
            blocked=[Tensor.backward,Base.step,Adam.step,AdamW.step,torch.autograd.backward,torch.autograd.grad,
                     agent.update,agent.learn,agent.optimizer.step,agent.surrogate_optimizer.step]
            for fn in blocked:
                with pytest.raises(RuntimeError,match='forbids'):fn()
            if exception:raise Deliberate()
    except Deliberate:pass
    assert before==(Tensor.backward,Base.step,Adam.step,AdamW.step,torch.autograd.backward,torch.autograd.grad,agent.update)
    assert not hasattr(agent,'learn')
    assert 'step' not in agent.optimizer.__dict__


def test_exclusive_claim_and_failed_run_no_runtime(tmp_path,monkeypatch):
    d.write(tmp_path/'contract.json',{'status':'frozen'})
    def verify(*a):raise ValueError('synthetic changed source')
    monkeypatch.setattr(d,'verify',verify)
    monkeypatch.setattr(d,'runtime',lambda *a:pytest.fail('must not load models'))
    with pytest.raises(ValueError,match='synthetic changed source'):d.run(SimpleNamespace(output_root=tmp_path))
    failure=d.read(tmp_path/'failure.json')
    assert failure['status']=='paused_failed'
    assert not any(failure['counts'].values())
    before=(tmp_path/'failure.json').read_bytes()
    with pytest.raises(FileExistsError):d.run(SimpleNamespace(output_root=tmp_path))
    assert (tmp_path/'failure.json').read_bytes()==before


def test_reference_parity_is_exact():
    head=dict(actual_mask=[True,False],valid_total_logits=[0.5],valid_learned_logits=[0.1],valid_public_prior=[0.4],greedy_action=0)
    reference={'heads':[head]};d.check_reference_parity(reference,deepcopy(reference))
    changed=deepcopy(reference);changed['heads'][0]['valid_total_logits'][0]+=1e-10
    with pytest.raises(ValueError):d.check_reference_parity(reference,changed)


def test_causal_physical_ledger_fake_env_only():
    from dataclasses import dataclass
    @dataclass
    class Costs:
        link_slots_per_rsu:int=1
        compute_slots_per_rsu:int=1
    class Workflow:
        workflow_id='known_active'
        def to_dict(self):return {'workflow_id':self.workflow_id,'remaining':['node0']}
    workflow=Workflow(); a=SimpleNamespace(request_id='v0/known_active',frame=0,workflow=workflow)
    env=SimpleNamespace(frame_index=0,frames=[{'time_index':999}],_active={'v0':a},_queues={'v0':[a]},
        workflows={'v0':workflow},rsus=[SimpleNamespace(rsu_id='r0')],costs=Costs(),
        model_cache=SimpleNamespace(snapshot=lambda rid:{'resident_object_ids':['base:0'],'occupied_bytes':2048}),
        input_received={('v0','n0','r0')},prepared_state={'v0':{'r1','r0'}},_released={'rB','rA'},
        arrivals='FORBIDDEN_FUTURE_INFORMATION')
    before=deepcopy(env.input_received); ledger=d.physical_state(env)
    assert env.input_received==before
    assert ledger['input_received']=={'__set_type__':'set','items':[('v0','n0','r0')]}
    assert ledger['prepared_state']['v0']['items']==['r0','r1']
    assert 'arrivals' not in ledger and 'FORBIDDEN_FUTURE_INFORMATION' not in repr(ledger)
    assert ledger['active']['v0']['request_id']=='v0/known_active'
    assert ledger['released_queues']['v0'][0]['workflow']==workflow.to_dict()
    assert ledger['rsu_cache']['r0']['occupied_bytes']==2048


def test_stable_set_order_across_hash_seeds():
    import os,subprocess,sys
    code="import importlib.util; s=importlib.util.spec_from_file_location('d',"+repr(str(SCRIPT))+"); d=importlib.util.module_from_spec(s); s.loader.exec_module(d); print(repr(d.stable_sets({'x':set(['bb','aa','cc']),('k',1):frozenset(['v2','v1'])})))"
    outputs=[subprocess.check_output([sys.executable,'-c',code],env=dict(os.environ,PYTHONHASHSEED=str(seed)),text=True) for seed in (1,2)]
    assert outputs[0]==outputs[1]
