"""Synthetic-only qualification for the prospective reward-curve runner."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import scripts.run_stream_reward_curve as runner
from tests.test_stream_control_learning import env


def plan(tmp_path):
    ids = ['case%02d' % i for i in range(12)]
    return dict(contract=runner.CONTRACT, source_commit='fixture', methods=runner.METHODS,
        seeds=runner.SEEDS, episodes_per_cell=96, update_group_size=4, update_epochs=4,
        training_case_ids=ids, training_schedule=ids * 8, probe_case_ids=ids[:3],
        checkpoint_episodes=runner.CHECKPOINT_EPISODES,
        outputs={m: {str(s): str((tmp_path / m / str(s)).resolve()) for s in runner.SEEDS}
                 for m in runner.METHODS},
        budgets=dict(maximum_frames_per_episode=480, maximum_training_frames_per_cell=46080,
            maximum_evaluation_frames_per_cell=23040, maximum_cell_wall_seconds=1000,
            maximum_cell_disk_bytes=503316480))


def test_fixed_balanced_plan_accepts(tmp_path):
    p = plan(tmp_path)
    assert runner.validate_plan(p, 'main', 7, tmp_path / 'main/7', verify_files=False) is p
    assert len(p['checkpoint_episodes']) == 13
    assert 12 * 3 + 12 == 48  # ep0 and eleven intermediate points, then final12.


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(episodes_per_cell=100),
    lambda p: p.update(update_epochs=8),
    lambda p: p.update(update_group_size=2),
    lambda p: p.update(seeds=[7]),
    lambda p: p.update(methods=['main']),
    lambda p: p['training_schedule'].__setitem__(0, p['training_case_ids'][1]),
    lambda p: p.update(probe_case_ids=['hidden', 'other', 'unseen']),
    lambda p: p.update(parent_checkpoint='winner.pt'),
    lambda p: p.update(checkpoint_episodes=[0, 96]),
    lambda p: p['budgets'].update(maximum_training_frames_per_cell=999999),
    lambda p: p['budgets'].update(maximum_cell_disk_bytes=999999999999),
])
def test_plan_rejects_drift_before_launch(tmp_path, mutation):
    p = plan(tmp_path); mutation(p)
    with pytest.raises(ValueError):
        runner.validate_plan(p, 'main', 7, tmp_path / 'main/7', verify_files=False)


def test_wrong_output_and_unknown_cell_rejected(tmp_path):
    p = plan(tmp_path)
    for method, seed, output in [('main', 7, tmp_path / 'elsewhere'), ('main', 9, tmp_path), ('best', 7, tmp_path)]:
        with pytest.raises(ValueError):
            runner.validate_plan(p, method, seed, output, verify_files=False)


def test_all_rng_restored_even_on_evaluation_error():
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    with pytest.raises(RuntimeError):
        with runner.isolated_rng():
            random.seed(51); np.random.seed(52); torch.manual_seed(53)
            random.random(); np.random.rand(); torch.rand(9)
            raise RuntimeError('probe failure')
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def make_traces(tmp_path):
    limits = plan(tmp_path)['budgets']
    budget = runner.CellBudget(tmp_path, limits)
    receipts = runner.GzipRecords(tmp_path / 'receipts.gz', budget)
    controls = runner.GzipRecords(tmp_path / 'controls.gz', budget)
    return budget, receipts, controls


@pytest.mark.parametrize('method', runner.METHODS)
def test_complete_native_tail_late_wave_likelihood_and_checkpoint(tmp_path, method):
    torch.set_num_threads(1)
    budget, receipts, controls = make_traces(tmp_path)
    agent = runner.build_policy(method, 7)
    e = env()
    identity = dict(method=method, seed=7, episode=1, phase='train', case_id='fixture')
    row, rolls = runner.physical_episode(e, agent, budget, 'train', identity, receipts, controls)
    receipts.close(); controls.close()
    frames = [json.loads(line) for line in gzip.open(tmp_path / 'receipts.gz', 'rt')]
    assert row['physical_terminal'] and row['bootstrap_team_tail'] == 0.
    assert 45 < row['steps'] <= 70 and budget.frames['train'] == row['steps']
    assert len(frames) == row['steps'] and len(e._released) == 3
    assert sum(r['reward'] for r in frames) == pytest.approx(row['reward'])
    assert rolls[-1]['episode_end']
    for r in rolls:
        observed = sum(f['reward'] for f in frames[r['frame_index']:]) / 10
        assert torch.allclose(r['returns'], torch.full_like(r['returns'], observed), atol=1e-6)
        assert torch.allclose(agent._evaluate(r['data'], actions=r['actions'])[1], r['head_logs'], atol=1e-6)
    record = runner.checkpoint_policy(agent, tmp_path / 'initial.pt',
        dict(method=method, seed=7, episode=0, cumulative_training_frames=0, cumulative_updates=0), budget)
    with runner.isolated_rng():
        loaded = runner.loaded_policy(tmp_path / 'initial.pt', record, type(agent))
    assert runner.tensor_hash(loaded) == runner.tensor_hash(agent)
    assert record['policy_initialization'] == 'random_from_scratch_no_policy_parent'
    assert record['predictor_initialization'].startswith('shared_frozen')


def test_idle_frame_receipts_no_padding_and_pre_step_frame_budget(tmp_path):
    from tests.test_multi_request_stream_env import make, wf
    from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv, WorkflowArrival
    from src.envs.specs import RSUState
    old = make([WorkflowArrival(8, 'v', wf('late'))], horizon=70)
    e = MultiRequestStreamEnv(frames=old.frames, arrivals=old.arrivals, catalog=old.catalog_objects,
        rsu_states=[RSUState(s, 0, 0, 10) for s in 'abcd'], costs=old.costs,
        workload_bytes_per_unit=1, shared_state_enabled=False, coordination_intent_enabled=True,
        link_reservation_enabled=True, model_cache_config={'capacity_bytes': 8, 'transfer_bytes_per_frame': 4})
    budget, receipts, controls = make_traces(tmp_path)
    row, rolls = runner.physical_episode(e, runner.build_policy('main', 7), budget, 'train', {}, receipts, controls)
    receipts.close(); controls.close()
    frames = [json.loads(s) for s in gzip.open(tmp_path / 'receipts.gz', 'rt')]
    assert len(frames) == e.frame_index < 70 and rolls[0]['frame_index'] == 8
    assert all(not f['receipt']['request_rewards'] for f in frames[:8])
    wrapped = runner.GuardedEnv(env(), budget, 'train')
    budget.frames['train'] = budget.limits['maximum_training_frames_per_cell']
    with pytest.raises(RuntimeError, match='frame budget'):
        wrapped.step_controls({})
    assert wrapped.frame_index == 0


def test_nonterminal_session_over_cap_rejected_before_step(tmp_path):
    budget = runner.CellBudget(tmp_path, dict(plan(tmp_path)['budgets'], maximum_frames_per_episode=50))
    e = env()
    with pytest.raises(ValueError, match='frame cap'):
        runner.GuardedEnv(e, budget, 'train')
    assert e.frame_index == 0


def test_compressed_disk_budget_checks_before_bytes(tmp_path):
    limits = dict(plan(tmp_path)['budgets'], maximum_cell_disk_bytes=65540)
    budget = runner.CellBudget(tmp_path, limits)
    writer = runner.LimitedBinaryWriter(tmp_path / 'trace', budget)
    writer.write(b'abcd')
    with pytest.raises(RuntimeError, match='disk budget'):
        writer.write(b'e')
    writer.close()
    assert (tmp_path / 'trace').read_bytes() == b'abcd'


def test_deadline_and_wall_checked_before_physical_step(tmp_path, monkeypatch):
    budget = runner.CellBudget(tmp_path, plan(tmp_path)['budgets'], deadline=0)
    e = env(); guarded = runner.GuardedEnv(e, budget, 'train')
    with pytest.raises(RuntimeError, match='global deadline'):
        guarded.step_controls({})
    assert e.frame_index == 0
    budget.deadline = None; budget.started = -1e99
    with pytest.raises(RuntimeError, match='wall budget'):
        guarded.step_controls({})
    assert e.frame_index == 0


def test_checkpoint_method_class_and_options_identity_rejected(tmp_path):
    budget = runner.CellBudget(tmp_path, plan(tmp_path)['budgets'])
    agent = runner.build_policy('mappo', 7)
    record = runner.checkpoint_policy(agent, tmp_path / 'm.pt', dict(method='mappo', seed=7), budget)
    corrupt = deepcopy(record); corrupt['method'] = 'ppo'
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.loaded_policy(tmp_path / 'm.pt', corrupt, type(agent))
    with pytest.raises(ValueError, match='class mismatch'):
        runner.loaded_policy(tmp_path / 'm.pt', record, type(runner.build_policy('ppo', 7)))


def test_evaluation_uses_loaded_policy_preserves_rng_and_training_controller(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    budget, receipts, controls = make_traces(tmp_path)
    agent = runner.build_policy('main', 7)
    agent.request_memory = {'untouched': 'training'}
    record = runner.checkpoint_policy(agent, tmp_path / 'main.pt', dict(method='main', seed=7,
        episode=8, cumulative_training_frames=999, cumulative_updates=2), budget)
    monkeypatch.setattr(runner, 'make_env', lambda *args: env())
    random.seed(18); np.random.seed(18); torch.manual_seed(18)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    random.seed(18); np.random.seed(18); torch.manual_seed(18)
    with (tmp_path / 'eval.jsonl').open('w') as output:
        rows = runner.evaluate_checkpoint(agent, record, ['a', 'b', 'c'], {'a': {}, 'b': {}, 'c': {}},
            {}, {}, budget, receipts, controls, output)
    receipts.close(); controls.close()
    assert agent.request_memory == {'untouched': 'training'}
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])
    assert [r['case_id'] for r in rows] == ['a', 'b', 'c']
    assert all(r['checkpoint_sha256'] == record['sha256'] and r['checkpoint_episode'] == 8 for r in rows)
    assert budget.frames['train'] == 0 and budget.frames['evaluation'] > 0


def test_failed_cell_preserves_claim_and_refuses_second_launch(tmp_path, monkeypatch):
    p = plan(tmp_path); campaign = tmp_path / 'campaign.json'; campaign.write_text(json.dumps(p))
    p['protocol'] = {'path': str(tmp_path / 'absent_manifest.json'), 'protocol_pt': 'absent.pt'}
    output = tmp_path / 'main/7'
    with pytest.raises(FileNotFoundError):
        runner.run_cell(p, campaign, 'main', 7, output, verify=False)
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['status'] == 'failed_preserved_do_not_retry'
    assert failure['training_frames'] == failure['evaluation_frames'] == 0
    old = (output / 'failure.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_cell(p, campaign, 'main', 7, output, verify=False)
    assert (output / 'failure.json').read_bytes() == old


@pytest.mark.parametrize('method', ['main', 'dqn'])
def test_native_optimizer_calls_observed_and_original_update_unchanged(tmp_path, method):
    torch.set_num_threads(1)
    agent = runner.build_policy(method, 7)
    from scripts.train_joint_stream_control import stream_episode
    rolls = []
    for _ in range(4):
        rolls.extend(stream_episode(env(), agent)[1])
    if method == 'dqn':
        # Replay's fixed native warm-up is 128 decisions, independent of rewards.
        while sum(len(r['actions']) for r in rolls) < 128:
            rolls.extend(stream_episode(env(), agent)[1])
    original = agent.optimizer.step
    losses, counts = runner.native_update(agent, rolls, 1)
    assert int(agent.completed_updates) == 1
    assert agent.optimizer.step == original
    assert counts['optimizer'] > 0
    if method == 'dqn':
        assert counts['optimizer'] == losses['gradient_steps'] == 8
    else:
        assert counts['surrogate_optimizer'] > 0


def test_full_synthetic_cell_has_exact_counts_fixed_probe_schedule_and_initial_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    p = plan(tmp_path)
    source_path = tmp_path / 'source.json'; source_path.write_text('{}')
    data_path = tmp_path / 'protocol.pt'
    selected = [dict(case_id=i, session='fixture_session') for i in p['training_case_ids']]
    torch.save(dict(cases={'train': selected}, sessions={'fixture_session': {'split': 'train'}}), data_path)
    p['protocol'] = dict(path=str(source_path), protocol_pt=str(data_path))
    campaign = tmp_path / 'campaign.json'; campaign.write_text(json.dumps(p))
    monkeypatch.setattr(runner, 'select_train_cases', lambda data: selected)
    monkeypatch.setattr(runner, 'make_env', lambda *args: env())
    output = tmp_path / 'mappo/7'
    result = runner.run_cell(p, campaign, 'mappo', 7, output, verify=False)
    assert result['training_episodes'] == 96 and result['evaluation_episodes'] == 48
    assert result['update_calls'] == 24 and result['gradient_steps'] > 0
    training = [json.loads(s) for s in (output / 'training.jsonl').read_text().splitlines()]
    evaluation = [json.loads(s) for s in (output / 'evaluation.jsonl').read_text().splitlines()]
    checkpoints = [json.loads(s) for s in (output / 'checkpoints.jsonl').read_text().splitlines()]
    assert [c['episode'] for c in checkpoints] == runner.CHECKPOINT_EPISODES
    assert checkpoints[0]['cumulative_training_frames'] == checkpoints[0]['cumulative_updates'] == 0
    assert [r['case_id'] for r in training] == p['training_schedule']
    assert result['training_frames'] == sum(r['steps'] for r in training)
    assert result['evaluation_frames'] == sum(r['steps'] for r in evaluation)
    for episode in runner.CHECKPOINT_EPISODES[:-1]:
        assert [r['case_id'] for r in evaluation if r['episode'] == episode] == p['probe_case_ids']
    final = [r for r in evaluation if r['episode'] == 96]
    assert [r['case_id'] for r in final] == p['training_case_ids']
    assert sum(r['is_curve_probe'] for r in final) == 3
    assert all(r['is_fixed_final'] for r in final)
    receipt_rows = [json.loads(s) for s in gzip.open(output / 'receipts.jsonl.gz', 'rt')]
    assert len(receipt_rows) == result['training_frames'] + result['evaluation_frames']
    assert (output / 'complete.json').exists() and not (output / 'failure.json').exists()
