"""Released intent, policy-only restrictions and matched behavior likelihoods."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from src.agents.sa_ghmappo_agent import (
    IntentAwareStreamSAGHMAPPOAgent as Intent,
    RelocationAwareStreamSAGHMAPPOAgent as Parent,
)
from tests.test_stream_control_learning import env


def test_disabled_intent_exact_parent_features_and_actions():
    torch.set_num_threads(1)
    old = Parent(); new = Intent(cache_intent_enabled=False)
    new.load_state_dict(old.state_dict())
    a, ra = old.act_controls(env().decision_states(), 0, deterministic=True)
    b, rb = new.act_controls(env().decision_states(), 0, deterministic=True)
    assert a == b
    for key in ra['data']:
        assert torch.equal(ra['data'][key], rb['data'][key]), key
    assert torch.equal(ra['head_logs'], rb['head_logs'])


def test_intent_requires_matching_released_peer_object_or_feasible_handoff():
    p = Intent(); decision = env().decision_states()['v']
    state = decision['semantic_state']; node = state['current_workflow_node']['node_id']
    p._team_targets = {'v':'a'}; p._delivery_decisions = {'v':decision}
    p._feature_opportunities = {}; p._cache_values = {}
    assert p._cache_intent_support(decision, 'a', node, 'adapter') == 'own_target'
    assert p._cache_intent_support(decision, 'b', node, 'adapter') is None
    peer = deepcopy(decision); peer['semantic_state']['primary_vehicle_id'] = 'peer'
    p._team_targets['peer'] = 'b'; p._delivery_decisions['peer'] = peer
    assert p._cache_intent_support(decision, 'b', node, 'adapter') == 'peer_released_demand'
    for w in [peer['semantic_state']['workflow'], *[q['workflow'] for q in peer['semantic_state']['queued_workflows']]]:
        for n in w['nodes']:
            n['required_adapter'] = 'different'
    assert p._cache_intent_support(decision, 'b', node, 'adapter') is None
    # Sharing a station alone is insufficient; the peer's exact adapter matters.
    p._feature_opportunities = {'b':dict(prepare_feasible=True, cache_net_reward_proxy=1., eta_frames=2)}
    p._cache_values = {('b',node,'adapter'):dict(use_after_frames=3)}
    assert p._cache_intent_support(decision, 'b', node, 'adapter') == 'feasible_handoff'
    p._feature_opportunities['b']['eta_frames'] = 4
    assert p._cache_intent_support(decision, 'b', node, 'adapter') is None


def test_intent_only_removes_optional_slots_and_preserves_reactive_loading():
    old = Parent(); new = Intent(); ds = env().decision_states()
    for p in [old,new]:
        p._team_targets = {'v':'a'}; p._delivery_decisions = ds
    a, candidates, *_ = old._coordinated_features(ds['v'], 0, {})
    b, *_ = new._coordinated_features(ds['v'], 0, {})
    assert not np.any(b['m1'] & ~a['m1'])
    for key in a:
        if key not in ('m1','f1'):
            assert np.array_equal(a[key], b[key]), key
    current = ds['v']['semantic_state']['current_workflow_node']['node_id']
    for i,(site,nid,kind) in enumerate(candidates[1]):
        if site == 'a' and nid == current:
            assert a['m1'][i] == b['m1'][i]
    assert b['m1'][0]


def test_new_checkpoint_rejects_old_contract_and_warm_starts_identically(tmp_path):
    old = Parent(); old.completed_updates.fill_(8)
    path = tmp_path/'parent.pt'; old.save_checkpoint(path)
    a = Intent.from_relocation_checkpoint(path, training_budget=12)
    b = Intent.from_relocation_checkpoint(path, training_budget=12, prediction_enabled=False)
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    assert not a.optimizer.state and not a.surrogate_optimizer.state and a.completed_updates == 0
    with pytest.raises(ValueError, match='incompatible'):
        Intent.from_checkpoint(path)
    target = tmp_path/'new.pt'; a.save_checkpoint(target)
    q = Intent.from_checkpoint(target)
    assert q.cache_intent_enabled
    assert q.act_controls(env().decision_states(),0,deterministic=True)[0] == a.act_controls(env().decision_states(),0,deterministic=True)[0]


@pytest.mark.parametrize('options',[dict(cache_intent_enabled=False),dict(),
    dict(prediction_enabled=False),dict(optional_cache_enabled=False)])
def test_stored_policy_probabilities_and_fixed_budget_update(options, monkeypatch, tmp_path):
    import scripts.run_stream_main_ablation as runner
    monkeypatch.setattr(runner, 'make_env', lambda *args:env())
    p = Intent(**options)
    with (tmp_path/'trace').open('w') as trace:
        row, rolls = runner.train_unit({},None,None,p,8,trace,{})
    assert row['actual_frames'] == 8
    for r in rolls:
        assert torch.allclose(p._evaluate(r['data'],actions=r['actions'])[1],r['head_logs'],atol=1e-6)
    assert np.isfinite(p.update(rolls,epochs=1)['policy_loss'])
