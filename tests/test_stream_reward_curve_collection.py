"""Independent, synthetic-only consumer qualification; one native cell is reused.

Tamper cases refresh producer hashes deliberately: they exercise semantic audit
checks rather than merely detecting a changed compressed file's bytes.
"""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.collect_stream_reward_curve as collector
import scripts.plot_stream_reward_curve as plotter
import scripts.run_stream_reward_curve as runner
from scripts.plot_stream_reward_curve import grouped
from tests.test_stream_reward_curve import plan
from tests.test_stream_control_learning import env


def write_json(path, value):
    Path(path).write_text(runner.json_text(value))


def write_rows(path, values):
    Path(path).write_text(''.join(runner.json_text(value) for value in values))


def gzip_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def write_gzip_rows(path, values):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for value in values:
                stream.write(runner.json_text(value).encode())


def refresh_output_hashes(cell):
    value = json.loads((cell.home / 'complete.json').read_text())
    for name in value['output_sha256']:
        value['output_sha256'][name] = runner.file_hash(cell.home / name)
    write_json(cell.home / 'complete.json', value)


@pytest.fixture(scope='module')
def native_cell(tmp_path_factory):
    """A complete 96-training/48-evaluation software fixture, built exactly once."""
    torch.set_num_threads(1)
    existing = os.environ.get('PPO_MEC_REWARD_CURVE_FIXTURE_ROOT')
    if existing:
        root = Path(existing).resolve()
        campaign = root / 'campaign.json'
        prospective = json.loads(campaign.read_text())
        home = root / 'mappo/7'
        complete = json.loads((home / 'complete.json').read_text())
        assert complete['status'] == 'complete'
        assert complete['training_episodes'] == 96
        assert complete['evaluation_episodes'] == 48
        return SimpleNamespace(plan=prospective, campaign=campaign, home=home,
                               original={path: path.read_bytes()
                                         for path in home.rglob('*') if path.is_file()})
    root = tmp_path_factory.mktemp('reward_curve_consumer')
    prospective = plan(root)
    source = root / 'source.json'
    source.write_text('{}')
    data_path = root / 'protocol.pt'
    selected = [dict(case_id=case_id, session='fixture_session')
                for case_id in prospective['training_case_ids']]
    torch.save(dict(cases={'train': selected},
                    sessions={'fixture_session': {'split': 'train'}}), data_path)
    prospective['protocol'] = dict(path=str(source), protocol_pt=str(data_path))
    campaign = root / 'campaign.json'
    write_json(campaign, prospective)
    home = root / 'mappo/7'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, 'select_train_cases', lambda data: selected)
        patch.setattr(runner, 'make_env', lambda *args: env())
        complete = runner.run_cell(prospective, campaign, 'mappo', 7, home,
                                   verify=False)
    assert complete['training_episodes'] == 96
    assert complete['evaluation_episodes'] == 48
    return SimpleNamespace(plan=prospective, campaign=campaign, home=home,
                           original={path: path.read_bytes()
                                     for path in home.rglob('*') if path.is_file()})


@contextmanager
def restored(cell):
    """Restore the same fixture; no tamper causes additional training or replay."""
    try:
        yield
    finally:
        for path in cell.home.rglob('*'):
            if path.is_file() and path not in cell.original:
                path.unlink()
        for path, content in cell.original.items():
            if not path.exists() or path.read_bytes() != content:
                path.write_bytes(content)


def audit(cell, prospective=None):
    return collector.audit_cell(prospective or cell.plan, 'mappo', 7,
                                campaign_sha=runner.file_hash(cell.campaign))


def test_complete_native_cell_accepts_and_preserves_frame_axis(native_cell):
    result = audit(native_cell)
    training = collector.read_rows(native_cell.home / 'training.jsonl')
    evaluation = collector.read_rows(native_cell.home / 'evaluation.jsonl')
    assert result['training_frames'] == sum(row['steps'] for row in training)
    assert result['evaluation_frames'] == sum(row['steps'] for row in evaluation)
    assert result['update_calls'] == 24
    assert len(result['evaluation']) == 48
    assert len(result['checkpoints']) == 13
    assert len(result['frame_digests']) == 144
    assert len(result['control_digests']) == 144


@pytest.mark.parametrize('where', ['wrapper_reward', 'native_receipt_components'])
def test_reward_tamper_rejected_after_file_hash_refresh(native_cell, where):
    with restored(native_cell):
        path = native_cell.home / 'receipts.jsonl.gz'
        values = gzip_rows(path)
        if where == 'wrapper_reward':
            values[0]['reward'] += 1.
        else:
            parts = values[0]['receipt']['reward_components']
            component = next(iter(parts))
            parts[component] += 1.
        write_gzip_rows(path, values)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


@pytest.mark.parametrize('field', ['returns', 'team_credit_targets', 'credit_targets'])
def test_complete_mc_label_tamper_rejected(native_cell, field):
    with restored(native_cell):
        path = native_cell.home / 'controls.jsonl.gz'
        values = gzip_rows(path)
        row = next(value for value in values
                   if value['record_kind'] == 'decision_rollout')
        row[field] = torch.as_tensor(row[field]).add(1.).tolist()
        write_gzip_rows(path, values)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'terminal_flag'])
def test_mc_rollout_completeness_and_terminal_are_not_optional(native_cell, mutation):
    with restored(native_cell):
        path = native_cell.home / 'controls.jsonl.gz'
        values = gzip_rows(path)
        indices = [index for index, value in enumerate(values)
                   if value['record_kind'] == 'decision_rollout']
        if mutation == 'omit':
            values.pop(indices[0])
        elif mutation == 'duplicate':
            values.insert(indices[0], deepcopy(values[indices[0]]))
        else:
            last = next(value for value in reversed(values)
                        if value['record_kind'] == 'decision_rollout'
                        and value['episode_end'])
            last['episode_end'] = False
        write_gzip_rows(path, values)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


@pytest.mark.parametrize('field,changed', [
    ('policy_class', 'DifferentPolicyClass'),
    ('native_method', 'ppo'),
    ('options', {'prediction_enabled': False}),
])
def test_self_consistent_wrong_checkpoint_identity_rejected(native_cell, field, changed):
    with restored(native_cell):
        records_path = native_cell.home / 'checkpoints.jsonl'
        records = collector.read_rows(records_path)
        record = records[0]
        checkpoint = Path(record['path'])
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        record[field] = changed
        payload['reward_curve_identity'][field] = changed
        if field == 'options':
            payload['options'] = changed
        torch.save(payload, checkpoint)
        record['sha256'] = runner.file_hash(checkpoint)
        write_rows(records_path, records)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


def test_initial_weights_must_match_declared_fresh_seed_not_just_identity(native_cell):
    with restored(native_cell):
        path = native_cell.home / 'checkpoints.jsonl'
        records = collector.read_rows(path)
        record = records[0]
        checkpoint = Path(record['path'])
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        first = next(iter(payload['state_dict']))
        payload['state_dict'][first] = payload['state_dict'][first] + 1.
        torch.save(payload, checkpoint)
        record['sha256'] = runner.file_hash(checkpoint)
        record['tensor_sha256'] = hashlib.sha256(b''.join(
            tensor.numpy().tobytes() for tensor in payload['state_dict'].values())).hexdigest()
        write_rows(path, records)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


def test_campaign_digest_and_embedded_prospective_plan_bound(native_cell):
    with pytest.raises(ValueError):
        collector.audit_cell(native_cell.plan, 'mappo', 7, campaign_sha='0' * 64)
    with restored(native_cell):
        path = native_cell.home / 'protocol.json'
        protocol = json.loads(path.read_text())
        protocol['plan']['probe_case_ids'] = protocol['plan']['training_case_ids'][3:6]
        write_json(path, protocol)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


def test_missing_seed_from_declared_pool_rejected(native_cell):
    prospective = deepcopy(native_cell.plan)
    prospective['seeds'] = [7]
    with pytest.raises(ValueError):
        audit(native_cell, prospective)


def test_missing_fixed_policy_point_rejected_after_hash_refresh(native_cell):
    with restored(native_cell):
        path = native_cell.home / 'evaluation.jsonl'
        values = collector.read_rows(path)
        values.pop(3)  # One episode8 fixed probe is missing.
        write_rows(path, values)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


@pytest.mark.parametrize('confusion', ['curve_flag', 'final_flag', 'final_pool'])
def test_fixed_final_and_fixed_probe_never_interchanged(native_cell, confusion):
    with restored(native_cell):
        path = native_cell.home / 'evaluation.jsonl'
        values = collector.read_rows(path)
        if confusion == 'curve_flag':
            values[-1]['is_curve_probe'] = True
        elif confusion == 'final_flag':
            values[0]['is_fixed_final'] = True
        else:
            values = [row for row in values if row['episode'] != 96
                      or row['case_id'] in native_cell.plan['probe_case_ids']]
        write_rows(path, values)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


def test_required_receipt_inventory_cannot_be_omitted(native_cell):
    with restored(native_cell):
        path = native_cell.home / 'complete.json'
        complete = json.loads(path.read_text())
        del complete['output_sha256']['controls.jsonl.gz']
        write_json(path, complete)
        with pytest.raises(ValueError):
            audit(native_cell)


def test_observed_update_and_resource_bill_cannot_be_faked(native_cell):
    with restored(native_cell):
        path = native_cell.home / 'updates.jsonl'
        values = collector.read_rows(path)
        values[0]['observed_gradient_steps'] += 1
        write_rows(path, values)
        refresh_output_hashes(native_cell)
        with pytest.raises(ValueError):
            audit(native_cell)


@pytest.mark.parametrize('mutation', ['drop_seed', 'drop_point'])
def test_plot_grid_refuses_missing_seeds_or_checkpoints(mutation):
    seeds, episodes = [7, 17, 27, 37, 47], runner.CHECKPOINT_EPISODES
    points = [dict(seed=seed, episode=episode, reward=float(episode))
              for seed in seeds for episode in episodes]
    if mutation == 'drop_seed':
        points = [row for row in points if row['seed'] != 47]
    else:
        points.pop()
    with pytest.raises(ValueError):
        grouped(points, seeds, episodes)


def synthetic_plot_source(tmp_path):
    """Small synthetic audit tables test plot-to-audit binding without models."""
    cases = ['case%02d' % index for index in range(12)]
    probes = cases[:3]
    cells, series = [], []
    for method_index, method in enumerate(runner.METHODS):
        seed_returns, seed_frames = [], []
        for seed in runner.SEEDS:
            evaluation = []
            for episode in runner.CHECKPOINT_EPISODES:
                for case_index, case_id in enumerate(cases if episode == 96 else probes):
                    evaluation.append(dict(episode=episode, case_id=case_id,
                        reward=method_index * 10 + seed * .01 + episode * .2 + case_index))
            checkpoints = [dict(episode=episode, cumulative_training_frames=episode * 100)
                           for episode in runner.CHECKPOINT_EPISODES]
            cells.append(dict(method=method, seed=seed, evaluation=evaluation,
                              checkpoints=checkpoints))
            seed_returns.append([sum(row['reward'] for row in evaluation
                if row['episode'] == episode and row['case_id'] in probes) / 3
                for episode in runner.CHECKPOINT_EPISODES])
            seed_frames.append([episode * 100 for episode in runner.CHECKPOINT_EPISODES])
        series.append(dict(method=method, seed_returns=seed_returns, seed_frames=seed_frames))
    rules = [dict(case_id=case_id, reward=float(index)) for index, case_id in enumerate(cases)]
    audit_path = tmp_path / 'audit.json'
    write_json(audit_path, dict(status='curve_receipts_and_checkpoints_audited',
        cells=cells, probe_case_ids=probes, rules=rules))
    payload = dict(contract='stream_reward_curve_plot_v1', selection='fixed_final_no_peak_selection',
        audit_path=str(audit_path), audit_sha256=runner.file_hash(audit_path),
        seeds=runner.SEEDS, episodes=runner.CHECKPOINT_EPISODES,
        series=series, gain_rule_probe_mean=1.)
    path = tmp_path / 'plot_source.json'
    write_json(path, payload)
    return path, payload


def test_fixed_plot_accepts_bound_synthetic_points(tmp_path, monkeypatch):
    path, _ = synthetic_plot_source(tmp_path)
    marker = tmp_path / 'qualified.png'
    def fake_export(figure, *args, extra_figures=()):
        plotter.plt.close(figure)
        for _, extra in extra_figures:
            plotter.plt.close(extra)
        return marker
    monkeypatch.setattr(plotter, 'export', fake_export)
    assert plotter.fixed_policy(path, tmp_path / 'figure') == marker


@pytest.mark.parametrize('mutation', [
    'reward', 'frames', 'rule', 'final_full_pool_mean', 'missing_seed', 'missing_point',
])
def test_plotted_values_cannot_escape_the_native_audit(tmp_path, mutation):
    path, payload = synthetic_plot_source(tmp_path)
    if mutation == 'reward':
        payload['series'][0]['seed_returns'][0][0] += 1.
    elif mutation == 'frames':
        payload['series'][0]['seed_frames'][0][1] += 1
    elif mutation == 'rule':
        payload['gain_rule_probe_mean'] += 1.
    elif mutation == 'final_full_pool_mean':
        # The12-condition final mean must not replace the same3-probe curve point.
        audit = json.loads(Path(payload['audit_path']).read_text())
        final = [row['reward'] for row in audit['cells'][0]['evaluation']
                 if row['episode'] == 96]
        payload['series'][0]['seed_returns'][0][-1] = sum(final) / len(final)
    elif mutation == 'missing_seed':
        payload['seeds'] = payload['seeds'][:-1]
    else:
        payload['series'][0]['seed_returns'][0].pop()
    write_json(path, payload)
    try:
        with pytest.raises(ValueError):
            plotter.fixed_policy(path, tmp_path / 'figure')
    finally:
        plotter.plt.close('all')
