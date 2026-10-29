"""Synthetic qualification for one opt-in head-balanced main candidate.

No science data, reference campaign, sealed split or old audit is executed.
"""
from copy import deepcopy
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.agents.sa_ghmappo_agent import (IntentAwareStreamSAGHMAPPOAgent,
    HeadBalancedIntentStreamSAGHMAPPOAgent)
import scripts.run_stream_reward_curve as shared
import scripts.run_stream_head_balance_study as runner
import scripts.collect_stream_head_balance_study as collector
from scripts.train_joint_stream_control import stream_episode
from tests.test_stream_control_learning import env


def plan(tmp_path):
    ids=['train_000','train_014','train_019','train_007','train_024','train_038',
         'train_043','train_031','train_048','train_062','train_067','train_055']
    return dict(contract=runner.CONTRACT,source_commit='fixture',methods=runner.METHODS,
        seeds=runner.SEEDS,episodes_per_cell=96,update_group_size=4,update_epochs=4,
        checkpoint_episodes=runner.CHECKPOINT_EPISODES,training_case_ids=ids,
        training_schedule=ids*8,probe_case_ids=['train_000','train_038','train_067'],
        outputs={runner.METHOD:{str(s):str((tmp_path/'cells'/str(s)).resolve()) for s in runner.SEEDS}},
        root=str(tmp_path),workers=2,maximum_training_frames=230400,
        maximum_evaluation_frames=115200,maximum_replay_frames=1440,
        maximum_total_physical_frames=347040,maximum_wall_seconds=86400,
        maximum_disk_bytes=3221225472,global_deadline_unix=1e20,
        replay_episodes=deepcopy(runner.REPLAYS),acceptance=deepcopy(runner.ACCEPTANCE),
        budgets=dict(maximum_frames_per_episode=480,maximum_training_frames_per_cell=46080,
            maximum_evaluation_frames_per_cell=23040,maximum_cell_wall_seconds=1000,
            maximum_cell_disk_bytes=503316480))


def test_default_reduction_has_bitwise_original_values_and_gradients():
    agent=IntentAwareStreamSAGHMAPPOAgent(seed=7)
    objective=torch.randn(11,3,requires_grad=True);entropy=torch.rand(11,3,requires_grad=True)
    weight=torch.tensor([[1.,0.,1.]]*6+[[0.,1.,1.]]*5)
    actual=agent._reduce_active_policy_terms(objective,entropy,weight)
    expected=(-(objective*weight).sum()/weight.sum().clamp(min=1),
              (entropy*weight).sum()/weight.sum().clamp(min=1))
    assert all(torch.equal(x,y) for x,y in zip(actual,expected))
    assert all(torch.equal(x,y) for x,y in zip(
        torch.autograd.grad(sum(actual),(objective,entropy),retain_graph=True),
        torch.autograd.grad(sum(expected),(objective,entropy))))
    _,actual_penalty=agent._reduce_active_policy_terms(objective,entropy,weight,
                                                     entropy_coefficient=.01)
    expected_penalty=.01*(entropy*weight).sum()/weight.sum().clamp(min=1)
    assert torch.equal(actual_penalty,expected_penalty)


def test_rare_head_gradient_has_equal_total_head_weight():
    agent=runner.build_policy(runner.METHOD,7)
    objective=torch.randn(100,3,requires_grad=True);entropy=torch.rand(100,3,requires_grad=True)
    weight=torch.zeros(100,3);weight[:10,0]=1;weight[:1,1]=1;weight[:,2]=1
    policy,ent=agent._reduce_active_policy_terms(objective,entropy,weight)
    g=torch.autograd.grad(policy,objective)[0]
    assert torch.allclose(-g.sum(0),torch.full((3,),1/3))
    assert g[0,1].item()==pytest.approx(-1/3)
    assert g[0,2].item()==pytest.approx(-1/300)
    batch=agent._head_balance_batches[-1]
    assert batch['active_counts']==[10.,1.,100.]
    assert batch['objective_head_weights']==pytest.approx([1/3]*3)


def test_no_active_head_is_finite_zero_with_zero_gradient():
    agent=runner.build_policy(runner.METHOD,7)
    a=torch.randn(5,3,requires_grad=True);e=torch.randn(5,3,requires_grad=True)
    p,h=agent._reduce_active_policy_terms(a,e,torch.zeros(5,3))
    assert p.item()==h.item()==0
    gradients=torch.autograd.grad(p+h,(a,e))
    assert all(torch.count_nonzero(g)==0 for g in gradients)
    assert agent._head_balance_batches[-1]['objective_head_weights']==[0.,0.,0.]


def test_single_active_head_gets_unit_weight():
    agent=runner.build_policy(runner.METHOD,7)
    w=torch.zeros(8,3);w[:2,1]=1
    p,h=agent._reduce_active_policy_terms(torch.ones(8,3),torch.ones(8,3),w)
    assert p.item()==-1 and h.item()==1
    assert agent._head_balance_batches[-1]['objective_head_weights']==[0.,1.,0.]


@pytest.mark.parametrize('seed',runner.SEEDS)
def test_same_seed_initial_parameters_controls_and_probabilities_unchanged(seed):
    torch.set_num_threads(1)
    original=IntentAwareStreamSAGHMAPPOAgent(seed=seed,training_budget=96,return_gamma=1.,prediction_enabled=True)
    new=runner.build_policy(runner.METHOD,seed)
    assert original.state_dict().keys()==new.state_dict().keys()
    assert all(torch.equal(x,new.state_dict()[k]) for k,x in original.state_dict().items())
    before=env().decision_states();after=deepcopy(before)
    torch.manual_seed(99);a,roll_a=original.act_controls(before,0)
    torch.manual_seed(99);b,roll_b=new.act_controls(after,0)
    assert a==b
    assert torch.equal(roll_a['head_logs'],roll_b['head_logs'])
    assert torch.equal(roll_a['active_heads'],roll_b['active_heads'])
    assert new.options['policy_head_reduction']==new.head_reduction


@pytest.mark.parametrize('method',['main','mappo','ppo'])
def test_historical_full_update_bitwise_against_frozen_prechange_implementation(monkeypatch,method):
    import ast
    import subprocess
    import src.agents.ppo_agent as implementation
    torch.set_num_threads(1)
    # Read-only legacy source, not legacy data, checkpoints, audit or science.
    source=subprocess.check_output(['git','show',
        'e00d5faf9c3f73a689174237245ed6f85394b2f2:src/agents/ppo_agent.py'],text=True)
    module=ast.parse(source)
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='ScheduledControlPPO')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_update','exec'),
         implementation.__dict__,namespace)
    original=namespace['update']
    a=shared.build_policy(method,7)
    b=shared.build_policy(method,7)
    rolls=[]
    for _ in range(4):rolls.extend(stream_episode(env(),a)[1])
    torch.manual_seed(73);left=a.update(deepcopy(rolls),epochs=1)
    with monkeypatch.context() as patch:
        patch.setattr(implementation.ScheduledControlPPO,'update',original)
        torch.manual_seed(73);right=b.update(deepcopy(rolls),epochs=1)
    assert left==right
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_new_checkpoint_class_contract_and_options_roundtrip(tmp_path):
    a=runner.build_policy(runner.METHOD,7);path=tmp_path/'new.pt';a.save_checkpoint(path)
    b=type(a).from_checkpoint(path)
    assert a.options==b.options
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    with pytest.raises(ValueError,match='incompatible'):
        IntentAwareStreamSAGHMAPPOAgent.from_checkpoint(path)
    with pytest.raises(ValueError):
        HeadBalancedIntentStreamSAGHMAPPOAgent(policy_head_reduction='arbitrary')


def test_frozen_plan_accepts(tmp_path):
    p=plan(tmp_path)
    assert runner.validate_plan(p,runner.METHOD,7,p['outputs'][runner.METHOD]['7'],verify_files=False) is p


@pytest.mark.parametrize('change',[
    lambda p:p.update(methods=['main']),lambda p:p.update(seeds=[7]),
    lambda p:p.update(workers=3),lambda p:p.update(episodes_per_cell=192),
    lambda p:p.update(update_epochs=5),lambda p:p.update(checkpoint_episodes=[0,96]),
    lambda p:p['training_schedule'].__setitem__(0,'train_014'),
    lambda p:p.update(resume_checkpoint='best.pt'),
    lambda p:p.update(maximum_total_physical_frames=1000000),
    lambda p:p.update(maximum_wall_seconds=172800),
    lambda p:p.update(maximum_disk_bytes=4*1024**3),
    lambda p:p['budgets'].update(maximum_cell_disk_bytes=2**40),
    lambda p:p['replay_episodes'].__setitem__(0,dict(method=runner.METHOD,seed=7,case_id='train_000')),
    lambda p:p['acceptance'].update(minimum_nonregressing_seeds=1),
    lambda p:p['outputs'][runner.METHOD].update({'17':p['outputs'][runner.METHOD]['7']}),
])
def test_plan_rejects_scientific_drift(tmp_path,change):
    p=plan(tmp_path);change(p)
    with pytest.raises(ValueError):runner.validate_plan(p,runner.METHOD,7,p['outputs'][runner.METHOD]['7'],verify_files=False)


def test_new_cell_failure_preserved_no_second_launch(tmp_path):
    p=plan(tmp_path);p['protocol']={'path':str(tmp_path/'absent'),'protocol_pt':'absent'}
    path=tmp_path/'campaign.json';path.write_text(json.dumps(p));home=Path(p['outputs'][runner.METHOD]['7'])
    with pytest.raises(FileNotFoundError):runner.run_cell(p,path,runner.METHOD,7,home,verify=False)
    failure=(home/'failure.json').read_bytes()
    assert json.loads(failure)['training_frames']==0
    with pytest.raises(FileExistsError):runner.run_cell(p,path,runner.METHOD,7,home,verify=False)
    assert (home/'failure.json').read_bytes()==failure


@pytest.fixture(scope='module')
def synthetic_cell(tmp_path_factory):
    torch.set_num_threads(1)
    existing=os.environ.get('PPO_MEC_HEAD_BALANCE_FIXTURE_ROOT')
    if existing:
        root=Path(existing).resolve();path=root/'campaign.json';p=json.loads(path.read_text())
        home=Path(p['outputs'][runner.METHOD]['7'])
        complete=json.loads((home/'complete.json').read_text())
        assert complete['training_episodes']==96 and complete['evaluation_episodes']==48
        return SimpleNamespace(plan=p,path=path,home=home,
            original={f:f.read_bytes() for f in home.rglob('*') if f.is_file()})
    root=tmp_path_factory.mktemp('head_balance_synthetic_cell');p=plan(root)
    source=root/'source.json';source.write_text('{}');data=root/'protocol.pt'
    selected=[dict(case_id=cid,session='fixture_session') for cid in p['training_case_ids']]
    torch.save(dict(cases={'train':selected},sessions={'fixture_session':{'split':'train'}}),data)
    p['protocol']=dict(path=str(source),protocol_pt=str(data));path=root/'campaign.json';path.write_text(runner.json_text(p))
    home=Path(p['outputs'][runner.METHOD]['7'])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner,'select_train_cases',lambda data:selected)
        patch.setattr(runner,'make_env',lambda *args:env())
        patch.setattr(shared,'make_env',lambda *args:env())
        result=runner.run_cell(p,path,runner.METHOD,7,home,verify=False)
    assert result['training_episodes']==96 and result['evaluation_episodes']==48
    return SimpleNamespace(plan=p,path=path,home=home,original={f:f.read_bytes() for f in home.rglob('*') if f.is_file()})


@contextmanager
def restore(cell):
    try:yield
    finally:
        for p,value in cell.original.items():
            if not p.exists() or p.read_bytes()!=value:p.write_bytes(value)


def update_rows(cell,rows):
    (cell.home/'updates.jsonl').write_text(''.join(runner.json_text(x) for x in rows))
    complete=json.loads((cell.home/'complete.json').read_text())
    complete['output_sha256']['updates.jsonl']=runner.file_hash(cell.home/'updates.jsonl')
    (cell.home/'complete.json').write_text(runner.json_text(complete))


def test_full_native_synthetic_cell_mc_control_and_head_ledger_pass(synthetic_cell):
    c=synthetic_cell
    result=collector.audit_cell(c.plan,runner.METHOD,7,runner.file_hash(c.path))
    assert len(result['evaluation'])==48 and len(result['checkpoints'])==13
    assert len(result['head_update_diagnostics'])==24
    assert all(x['complete_epochs']==4 or x['kl_early_stop'] for x in result['head_update_diagnostics'])
    assert result['training_frames']>0


@pytest.mark.parametrize('kind',['count','weight','index','policy','reduction','omit','dtype'])
def test_consumer_rejects_semantic_head_tamper_after_hash_refresh(synthetic_cell,kind):
    c=synthetic_cell
    with restore(c):
        rows=collector.read_rows(c.home/'updates.jsonl');loss=rows[0]['native_losses'];batch=loss['head_balance_batches'][0]
        if kind=='count':batch['active_counts'][0]+=1
        elif kind=='weight':batch['objective_head_weights'][0]=1.
        elif kind=='index':batch['batch_indices'][1]=batch['batch_indices'][0]
        elif kind=='policy':batch['policy_loss']+=.1
        elif kind=='reduction':loss['policy_head_reduction']='legacy'
        elif kind=='dtype':batch['reduction_dtype']='torch.float64'
        else:loss['head_balance_batches'].pop()
        update_rows(c,rows)
        with pytest.raises(ValueError):collector.audit_cell(c.plan,runner.METHOD,7,runner.file_hash(c.path))


def fake_cells(reward_gain=1.,completion_delta=0.,state_delta=0.,negative_seeds=0):
    old=[];new=[]
    for i,s in enumerate(runner.SEEDS):
        base=[dict(episode=96,case_id='case'+str(j),reward=20.,completed_requests=16.,state_transfer_bytes=20.,state_transfer_link_frames=2.,handoff_failed_count=1.) for j in range(12)]
        changed=[dict(r,reward=r['reward']+(reward_gain if i>=negative_seeds else -1.),completed_requests=r['completed_requests']+completion_delta,state_transfer_bytes=r['state_transfer_bytes']+state_delta) for r in base]
        old.append(dict(seed=s,evaluation=base));new.append(dict(seed=s,evaluation=changed))
    return new,old


@pytest.mark.parametrize('kwargs,accepted',[({},True),({'reward_gain':0},False),({'completion_delta':-1},False),({'state_delta':1},False),({'negative_seeds':2},False),({'negative_seeds':1},True)])
def test_fixed_gate_reports_rejections_without_peak_or_seed_selection(kwargs,accepted):
    new,old=fake_cells(**kwargs);gate,rows=collector.paired_gate(new,old,runner.ACCEPTANCE)
    assert gate['accepted'] is accepted and len(rows)==60
    assert not gate['canonical_promoted'] and not gate['registry_promoted']


def test_collector_guards_deadline_disk_and_replay_before_step(tmp_path):
    p=plan(tmp_path);guard=collector.CollectionGuard(p)
    p['global_deadline_unix']=0
    with pytest.raises(RuntimeError,match='deadline'):guard.before_step()
    p['global_deadline_unix']=1e20;guard.replay_frames=1440
    with pytest.raises(RuntimeError,match='physical'):guard.before_step()
    guard.replay_frames=0;p['maximum_disk_bytes']=65536
    (tmp_path/'x').write_text('x')
    with pytest.raises(RuntimeError,match='disk'):guard.check()


@pytest.mark.parametrize('tamper',[None,'seed','tensor','reward','steps','frame','control'])
def test_initial_reference_requires_identical_planned_forward_probes(tamper):
    candidates=[];references=[];probes=['train_000','train_038','train_067']
    for seed in runner.SEEDS:
        cell=dict(seed=seed,checkpoints=[{'tensor_sha256':'initial'+str(seed)}],
            evaluation=[dict(episode=0,case_id=cid,reward=2.,steps=10) for cid in probes],
            frame_digests={'evaluation|0|'+cid:'frame' for cid in probes},
            control_digests={'evaluation|0|'+cid:'control' for cid in probes})
        references.append(cell);candidates.append(deepcopy(cell))
    c=candidates[0]
    if tamper=='seed':c['seed']=17
    elif tamper=='tensor':c['checkpoints'][0]['tensor_sha256']='different'
    elif tamper=='reward':c['evaluation'][0]['reward']+=1
    elif tamper=='steps':c['evaluation'][0]['steps']+=1
    elif tamper=='frame':c['frame_digests']['evaluation|0|train_000']='different'
    elif tamper=='control':c['control_digests']['evaluation|0|train_000']='different'
    if tamper:
        with pytest.raises(ValueError):collector.validate_initial_reference(candidates,references,probes)
    else:collector.validate_initial_reference(candidates,references,probes)


def test_new_policy_evaluation_rng_isolated_and_controller_unmodified(tmp_path,monkeypatch):
    import random
    torch.set_num_threads(1)
    p=plan(tmp_path);budget=runner.CellBudget(tmp_path,p['budgets'])
    receipts=runner.GzipRecords(tmp_path/'receipt.gz',budget)
    controls=runner.GzipRecords(tmp_path/'control.gz',budget)
    agent=runner.build_policy(runner.METHOD,7);agent.request_memory={'keep':'training'}
    cp=runner.checkpoint_policy(agent,tmp_path/'checkpoint.pt',
        dict(method=runner.METHOD,seed=7,episode=0,cumulative_training_frames=0,cumulative_updates=0),budget)
    monkeypatch.setattr(shared,'make_env',lambda *args:env())
    random.seed(101);np.random.seed(102);torch.manual_seed(103)
    expected=(random.random(),np.random.rand(),torch.rand(3))
    random.seed(101);np.random.seed(102);torch.manual_seed(103)
    with (tmp_path/'evaluation.jsonl').open('x') as output:
        rows=shared.evaluate_checkpoint(agent,cp,['fixture'],{'fixture':{}},{},{},budget,receipts,controls,output)
    receipts.close();controls.close()
    assert agent.request_memory=={'keep':'training'}
    assert random.random()==expected[0] and np.random.rand()==expected[1]
    assert torch.equal(torch.rand(3),expected[2])
    assert rows[0]['last_scheduled_release_frame']==45 and rows[0]['bootstrap_team_tail']==0


def fake_reference(tmp_path,monkeypatch):
    p=plan(tmp_path)
    original=dict(p,contract='stream_reward_curve_v1',methods=['main','mappo','ppo','dqn'],
        source_commit='e00d5faf9c3f73a689174237245ed6f85394b2f2')
    original['protocol']={};p['protocol']={}
    path=tmp_path/'old_plan.json';path.write_text(runner.json_text(original))
    cells=[dict(method=m,seed=seed,evaluation=[dict(seed=seed,episode=96,case_id=cid,
        reward=float(seed+i),completed_requests=16,state_transfer_bytes=42,
        state_transfer_link_frames=2,handoff_failed_count=1) for i,cid in enumerate(p['training_case_ids'])])
        for m in original['methods'] for seed in runner.SEEDS]
    old=dict(status='curve_receipts_and_checkpoints_audited',campaign_sha256=runner.file_hash(path),
        source_commit=original['source_commit'],cells=cells)
    audit=tmp_path/'old_audit.json';audit.write_text(runner.json_text(old))
    p['reference']=dict(campaign_path=str(path),campaign_sha256=runner.file_hash(path),
        audit_path=str(audit),audit_sha256=runner.file_hash(audit))
    monkeypatch.setattr(runner,'REFERENCE_CAMPAIGN_SHA',runner.file_hash(path))
    monkeypatch.setattr(runner,'REFERENCE_AUDIT_SHA',runner.file_hash(audit))
    qualification=dict(status='reference_final60_native_endpoints_qualified',
        campaign_sha256=runner.file_hash(path),audit_sha256=runner.file_hash(audit),
        source_commit=original['source_commit'],endpoints=[r for cell in cells if cell['method']=='main' for r in cell['evaluation']])
    qp=tmp_path/'endpoint_qualification.json';qp.write_text(runner.json_text(qualification))
    p['reference_endpoint_qualification']=dict(path=str(qp),sha256=runner.file_hash(qp))
    return p,old,audit,qualification,qp


def test_reference_is_hash_bound_complete_twenty_cell_readonly_control(tmp_path,monkeypatch):
    p,old,audit,qualification,qp=fake_reference(tmp_path,monkeypatch)
    assert len(runner.read_reference(p)[1]['cells'])==20
    old['cells'].pop();audit.write_text(runner.json_text(old))
    p['reference']['audit_sha256']=runner.file_hash(audit)
    monkeypatch.setattr(runner,'REFERENCE_AUDIT_SHA',runner.file_hash(audit))
    with pytest.raises(ValueError,match='twenty-cell'):runner.read_reference(p)


@pytest.mark.parametrize('tamper',['missing','hash','status','omit','duplicate','cost','episode','source'])
def test_reference_requires_exact_qualified_sixty_native_endpoints(tmp_path,monkeypatch,tamper):
    p,old,audit,q,path=fake_reference(tmp_path,monkeypatch)
    if tamper=='missing':p.pop('reference_endpoint_qualification')
    elif tamper=='hash':p['reference_endpoint_qualification']['sha256']='wrong'
    else:
        if tamper=='status':q['status']='partial'
        elif tamper=='omit':q['endpoints'].pop()
        elif tamper=='duplicate':q['endpoints'][0]=dict(q['endpoints'][1])
        elif tamper=='cost':q['endpoints'][0]['state_transfer_bytes']+=1
        elif tamper=='episode':q['endpoints'][0]['episode']=0
        else:q['source_commit']='different'
        path.write_text(runner.json_text(q));p['reference_endpoint_qualification']['sha256']=runner.file_hash(path)
    with pytest.raises(ValueError):runner.read_reference(p)


def test_guarded_large_json_fails_before_write(tmp_path):
    p=plan(tmp_path);p['maximum_disk_bytes']=65536+5
    guard=collector.CollectionGuard(p)
    with pytest.raises(RuntimeError,match='disk'):
        collector.save_guarded(tmp_path/'too_big.json',{'value':'large'},guard)
    assert not (tmp_path/'too_big.json').exists()


def test_pair_plot_keeps_all_seeds_all_points_and_three_probe_final(tmp_path):
    import csv
    p=plan(tmp_path);guard=collector.CollectionGuard(p);cells=[]
    for seed in runner.SEEDS:
        rows=[]
        for ep in runner.CHECKPOINT_EPISODES:
            ids=p['training_case_ids'] if ep==96 else p['probe_case_ids']
            rows.extend(dict(episode=ep,case_id=cid,is_curve_probe=cid in p['probe_case_ids'],
                reward=100.+seed+ep if cid in p['probe_case_ids'] else -10000.) for cid in ids)
        cells.append(dict(seed=seed,evaluation=rows,checkpoints=[dict(episode=ep,cumulative_training_frames=ep*10) for ep in runner.CHECKPOINT_EPISODES]))
    points=collector.export_curve(tmp_path,cells,deepcopy(cells),guard)
    assert len(points)==2*5*13
    assert {r['seed'] for r in points}==set(runner.SEEDS)
    assert all(r['reward']==100.+r['seed']+r['episode'] for r in points)
    assert all((tmp_path/('paired_reward_learning_curve.'+ext)).exists() for ext in ('pdf','svg','png'))
    assert len(list(csv.DictReader((tmp_path/'paired_curve_points.csv').open())))==130


def test_text_telemetry_is_charged_before_accepting_bytes(tmp_path):
    budget=runner.CellBudget(tmp_path,dict(plan(tmp_path)['budgets'],maximum_cell_disk_bytes=65540))
    with (tmp_path/'rows').open('x') as stream:
        writer=runner.BudgetedTextWriter(stream,budget);writer.write('abcd');writer.flush()
        with pytest.raises(RuntimeError,match='disk'):writer.write('x')
    assert (tmp_path/'rows').read_text()=='abcd'


def test_legitimate_large_float32_head_means_use_native_order_and_ulps(tmp_path):
    p=plan(tmp_path);agent=runner.build_policy(runner.METHOD,7)
    means=torch.tensor([[-60.123456,2.234567,1.345678]],dtype=torch.float32)
    value,_=agent._reduce_active_policy_terms(means,torch.full_like(means,.1),
        torch.ones_like(means),batch_indices=torch.tensor([0]))
    batch=deepcopy(agent._head_balance_batches[-1])
    reassociated=-float((np.asarray(batch['objective_head_means'],dtype=float)*(np.ones(3)/3)).sum())
    assert abs(float(value)-reassociated)>1e-6
    losses=dict(policy_head_reduction=agent.head_reduction,samples=1,gradient_steps=4,
        kl_early_stop=0,policy_loss=float(value),head_balance_batches=[deepcopy(batch) for _ in range(4)])
    updates=[dict(episode=4,native_losses=losses,
        optimizer_step_calls={'optimizer':4,'surrogate_optimizer':4})]
    rows={('train',ep,p['training_schedule'][ep-1]):([[True,True,True]] if ep==1 else []) for ep in range(1,5)}
    assert collector.audit_head_updates(p,updates,rows)[0]['complete_epochs']==4
    updates[0]['native_losses']['head_balance_batches'][0]['policy_loss']+=.1
    with pytest.raises(ValueError):collector.audit_head_updates(p,updates,rows)


@pytest.mark.parametrize('field',['state_transfer_bytes','state_transfer_link_frames',
                                  'handoff_failed_count','completed_requests','completed_nodes'])
def test_native_gate_summary_tamper_rejected_after_hash_refresh(synthetic_cell,field):
    c=synthetic_cell
    with restore(c):
        rows=collector.read_rows(c.home/'evaluation.jsonl')
        target=next(r for r in rows if r['episode']==96 and r['case_id']=='train_014')
        target[field]+=1
        path=c.home/'evaluation.jsonl';path.write_text(''.join(runner.json_text(r) for r in rows))
        complete=json.loads((c.home/'complete.json').read_text())
        complete['output_sha256'][path.name]=runner.file_hash(path)
        (c.home/'complete.json').write_text(runner.json_text(complete))
        with pytest.raises(ValueError,match='native|completed node'):
            collector.audit_cell(c.plan,runner.METHOD,7,runner.file_hash(c.path))


@pytest.mark.parametrize('field',['checkpoint_episode','cumulative_updates'])
def test_redundant_evaluation_checkpoint_identity_checked(synthetic_cell,field):
    c=synthetic_cell
    with restore(c):
        rows=collector.read_rows(c.home/'evaluation.jsonl');rows[0][field]+=1
        path=c.home/'evaluation.jsonl';path.write_text(''.join(runner.json_text(r) for r in rows))
        complete=json.loads((c.home/'complete.json').read_text());complete['output_sha256'][path.name]=runner.file_hash(path)
        (c.home/'complete.json').write_text(runner.json_text(complete))
        with pytest.raises(ValueError,match='checkpoint binding'):
            collector.audit_cell(c.plan,runner.METHOD,7,runner.file_hash(c.path))


def test_surrogate_optimizer_calls_equal_actual_head_minibatches(synthetic_cell):
    c=synthetic_cell
    with restore(c):
        rows=collector.read_rows(c.home/'updates.jsonl')
        rows[0]['optimizer_step_calls']['surrogate_optimizer']-=1
        rows[0]['observed_gradient_steps']-=1
        update_rows(c,rows)
        complete=json.loads((c.home/'complete.json').read_text());complete['gradient_steps']-=1
        (c.home/'complete.json').write_text(runner.json_text(complete))
        with pytest.raises(ValueError,match='minibatch'):
            collector.audit_cell(c.plan,runner.METHOD,7,runner.file_hash(c.path))
