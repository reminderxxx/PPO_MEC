"""Main-only prior intervention, unchanged legality, matched checkpoints and topology."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from src.agents.sa_ghmappo_agent import RelocationAwareStreamSAGHMAPPOAgent
from tests.test_stream_control_learning import env, policy


def candidate(**kwargs):
    torch.set_num_threads(1)
    return RelocationAwareStreamSAGHMAPPOAgent(**kwargs)


def test_neutral_mode_is_exact_original_and_parameter_count_equal():
    old=policy();new=candidate(relocation_cost_mode='episode')
    new.load_state_dict(old.state_dict())
    a,ra=old.act_controls(env().decision_states(),0,deterministic=True)
    b,rb=new.act_controls(env().decision_states(),0,deterministic=True)
    assert a==b
    for k in ra['data']:
        assert torch.equal(ra['data'][k],rb['data'][k]),k
    assert torch.equal(ra['head_logs'],rb['head_logs'])
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())


def test_prior_changes_only_restoration_term_and_never_action_masks():
    d=env().decision_states()['v'];s=d['semantic_state']
    s.update(execution_state_version=2,execution_state_payload_bytes=4,execution_location='a')
    # Make four RSU public states visible; no hidden future position is added.
    row=deepcopy(s['rsus'][0])
    for r in s['rsus']:
        r.update(occupied_bytes=0,capacity_bytes=8,resident_object_ids=[],loading_object_ids=[])
    old=candidate(relocation_cost_mode='episode');new=candidate()
    before=old._coordinated_features(d,0,{})[0]
    after=new._coordinated_features(d,0,{})[0]
    for k in before:
        if k!='prior0':assert np.array_equal(before[k],after[k]),k
    for i,site in enumerate([None,*sorted(r['rsu_id'] for r in s['rsus']),'vehicle']):
        if site=='a' or site not in s['reachable_execution_rsu_ids']:
            assert before['prior0'][i]==after['prior0'][i]
        else:assert after['prior0'][i]<before['prior0'][i]


def test_parent_warm_starts_share_weights_and_fresh_optimizers(tmp_path):
    old=policy();old.completed_updates.fill_(9)
    path=tmp_path/'parent.pt';old.save_checkpoint(path)
    full=RelocationAwareStreamSAGHMAPPOAgent.from_parent_checkpoint(path,training_budget=12)
    no_edges=RelocationAwareStreamSAGHMAPPOAgent.from_parent_checkpoint(path,training_budget=12,dag_topology_enabled=False)
    assert all(torch.equal(a,b) for a,b in zip(full.state_dict().values(),no_edges.state_dict().values()))
    assert full.completed_updates==0 and not full.optimizer.state and not full.surrogate_optimizer.state
    d=env().decision_states()['v']
    a=full._coordinated_features(d,0,{})[0];b=no_edges._coordinated_features(d,0,{})[0]
    assert a['adjacency'].any() and not b['adjacency'].any()
    assert np.array_equal(a['graph'],b['graph'])
    out=tmp_path/'new.pt';full.save_checkpoint(out)
    restored=RelocationAwareStreamSAGHMAPPOAgent.from_checkpoint(out)
    assert restored.relocation_cost_mode=='node'
    assert restored.act_controls(env().decision_states(),0,deterministic=True)[0]==full.act_controls(env().decision_states(),0,deterministic=True)[0]


def test_no_optional_cache_preserves_current_reactive_loading():
    p=candidate(optional_cache_enabled=False);ds=env().decision_states();p._team_targets={'v':'a'};p._delivery_decisions=ds
    data,cs,*_=p._coordinated_features(ds['v'],0,{})
    current=ds['v']['semantic_state']['current_workflow_node']['node_id']
    assert data['m1'][0]
    assert any(data['m1'][i] and site=='a' and nid==current for i,(site,nid,kind) in enumerate(cs[1]))
    assert all(not data['m1'][i] for i,(site,nid,kind) in enumerate(cs[1]) if nid and nid!=current)
    with pytest.raises(ValueError,match='main-method only'):
        candidate(method='mappo')


def test_fixed_budget_fragments_preserve_behavior_and_bootstrap_accounting(monkeypatch,tmp_path):
    import json
    import scripts.run_stream_main_ablation as runner
    monkeypatch.setattr(runner,'make_env',lambda *args:env())
    p=candidate()
    trace_path=tmp_path/'trace.jsonl'
    with trace_path.open('w') as trace:
        row,rolls=runner.train_unit({},None,None,p,73,trace,{'seed':7})
    trace=[json.loads(x) for x in trace_path.read_text().splitlines()]
    assert row['actual_frames']==len(trace)==73
    assert sum(f['frames'] for f in row['fragments'])==73
    assert row['reward']==pytest.approx(sum(x['reward'] for x in trace))
    assert row['fragments'][-1]['physical_terminal'] is False
    for r in rolls:
        _,logs,*_=p._evaluate(r['data'],actions=r['actions'])
        assert torch.allclose(logs,r['head_logs'],atol=1e-6)
        assert torch.allclose(r['credit_targets'],r['returns']/r['initial_request_count'])
    assert torch.isfinite(torch.cat([r['returns'] for r in rolls])).all()
    assert np.isfinite(p.update(rolls,epochs=1)['policy_loss'])


@pytest.mark.parametrize('variant',['original','relocation','no_dag_edges','no_prediction','no_surrogate','no_optional_cache'])
def test_every_study_variant_has_valid_stored_behavior_and_finite_update(variant,monkeypatch,tmp_path):
    import scripts.run_stream_main_ablation as runner
    monkeypatch.setattr(runner,'make_env',lambda *args:env())
    p=candidate(**runner.VARIANTS[variant])
    with (tmp_path/'trace').open('w') as trace:
        _,rolls=runner.train_unit({},None,None,p,8,trace,{})
    for r in rolls:
        assert torch.allclose(p._evaluate(r['data'],actions=r['actions'])[1],r['head_logs'],atol=1e-6)
    result=p.update(rolls,epochs=1)
    assert np.isfinite(result['policy_loss'])
