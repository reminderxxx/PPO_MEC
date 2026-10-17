"""Evaluation-only consumer tests; artifacts in tmp_path, no train/policy rollout."""
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from scripts import evaluate_joint_stream_learning_effectiveness as evaluate

SOURCE = ROOT / 'artifacts/joint_stream_learning/20261016T162100Z_confirmation'


def dump_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


@pytest.fixture
def real_frozen_reuse(tmp_path):
    if not SOURCE.is_dir():
        pytest.skip('Frozen confirmation artifacts are intentionally not tracked')
    copied = []
    provenance = {}
    for name in evaluate.NAMES:
        folder = SOURCE / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        provenance[name] = dict(manifest_sha256=evaluate.digest(folder / 'manifest.json'),
                                dev_sha256=evaluate.digest(folder / 'dev.jsonl'),
                                checkpoints=manifest['checkpoints'])
        copied.extend(dict(r, evaluation_origin='immutable_original')
                      for r in evaluate.read_rows(folder / 'dev.jsonl'))
    dump_rows(tmp_path / 'reused_rows.jsonl', copied)
    contract = dict(methods=evaluate.NAMES, seeds=evaluate.SEEDS, checkpoints=evaluate.POINTS,
                    new_training_sessions=0, new_cells=2016, reused_cells=1440,
                    workers=2, max_new_evaluation_frames=967680,
                    evaluator_sha256=evaluate.digest(evaluate.__file__),
                    source_run=str(SOURCE),
                    source_protocol_sha256=evaluate.digest(SOURCE / 'protocol/manifest.json'),
                    provenance=provenance,
                    reused_rows_sha256=evaluate.digest(tmp_path / 'reused_rows.jsonl'))
    evaluate.write_json(tmp_path / 'contract.json', contract)
    assert len(copied) == 1440
    return tmp_path, contract


def test_real_reused1440_rows_match_frozen_source(real_frozen_reuse):
    output, contract = real_frozen_reuse
    assert evaluate.verify_contract(output) == contract


def test_reused_reward_change_rejected_by_sha(real_frozen_reuse):
    output, _ = real_frozen_reuse
    rows = evaluate.read_rows(output / 'reused_rows.jsonl')
    rows[0]['reward'] += 1000
    dump_rows(output / 'reused_rows.jsonl', rows)
    with pytest.raises(AssertionError):
        evaluate.verify_contract(output)


def test_reused_reward_change_rejected_even_after_hash_refresh(real_frozen_reuse):
    output, contract = real_frozen_reuse
    rows = evaluate.read_rows(output / 'reused_rows.jsonl')
    rows[0]['reward'] += 1000
    dump_rows(output / 'reused_rows.jsonl', rows)
    contract['reused_rows_sha256'] = evaluate.digest(output / 'reused_rows.jsonl')
    evaluate.write_json(output / 'contract.json', contract)
    with pytest.raises(AssertionError):
        evaluate.verify_contract(output)


def test_duplicate_reused_identity_rejected(real_frozen_reuse):
    output, contract = real_frozen_reuse
    rows = evaluate.read_rows(output / 'reused_rows.jsonl')
    rows[1] = rows[0].copy()
    dump_rows(output / 'reused_rows.jsonl', rows)
    contract['reused_rows_sha256'] = evaluate.digest(output / 'reused_rows.jsonl')
    evaluate.write_json(output / 'contract.json', contract)
    with pytest.raises(AssertionError):
        evaluate.verify_contract(output)


def synthetic_summary_case(output, monkeypatch):
    ids = [f'dev_{i:03}' for i in range(48)]
    probes = ids[::4]
    contract = dict(panel_ids=ids, max_new_evaluation_frames=967680,
                    case_metadata={c: dict(session=3+int(c[-3:])//24,
                        size=['short', 'medium', 'long'][int(c[-3:]) % 3],
                        arrival_mode='stream', families=1, capacity=4096) for c in ids},
                    claim_limits=['synthetic consumer test'], provenance={})
    reused = []
    for method in evaluate.NAMES:
        checkpoints = []
        for seed in evaluate.SEEDS:
            checkpoints.append(dict(seed=seed, episode=32,
                initial_tensor_sha256=f'{method}:{seed}:initialTensor',
                initial_actor_sha256=f'{method}:{seed}:initialActor',
                trained_tensor_sha256=f'{method}:{seed}:trainedTensor',
                trained_actor_sha256=f'{method}:{seed}:trainedActor'))
        contract['provenance'][method] = {'checkpoints': checkpoints}
        for seed in evaluate.SEEDS:
            def row(episode, case_id, fresh):
                prefix = 'initial' if episode == 0 else 'trained'
                return dict(agent=method, seed=seed, episode=episode, case_id=case_id,
                    reward=float(episode + int(case_id[-3:])), request_count=16,
                    completed_requests=8, completed_nodes=80,
                    model_cache={'rsu_a': {'transferred_bytes': 1024, 'evicted_bytes': 512}},
                    state_transfer_bytes=256, service_target_change_count=4, steps=1,
                    evaluation_origin='fresh_deterministic' if fresh else 'immutable_original',
                    tensor_sha256=f'{method}:{seed}:{prefix}Tensor',
                    actor_sha256=f'{method}:{seed}:{prefix}Actor')
            reused.extend(row(ep, c, False) for ep in [32,64]
                          for c in (probes if ep == 32 else ids))
            fresh = [row(ep,c,True) for ep in [0,32] for c in ids
                     if ep == 0 or c not in probes]
            folder = output / f'{method}_seed{seed}'
            folder.mkdir()
            dump_rows(folder / 'rows.jsonl', fresh)
            evaluate.write_json(folder / 'result.json', dict(status='complete',
                new_cells_completed=84, evaluation_frames=84,
                output_sha256={'rows.jsonl':evaluate.digest(folder/'rows.jsonl')}))
    dump_rows(output / 'reused_rows.jsonl', reused)
    evaluate.write_json(output / 'contract.json', contract)
    monkeypatch.setattr(evaluate, 'verify_contract', lambda _: contract)
    return contract


def refresh_group_rows(output, mutation):
    folder = output / 'sa_ghmappo_seed7'
    rows = evaluate.read_rows(folder / 'rows.jsonl')
    mutation(rows)
    dump_rows(folder / 'rows.jsonl', rows)
    meta = json.loads((folder / 'result.json').read_text())
    meta['output_sha256']['rows.jsonl'] = evaluate.digest(folder / 'rows.jsonl')
    evaluate.write_json(folder / 'result.json',meta)


def test_summary_fixed48panel_0_32_64_alignment_and_metrics(tmp_path, monkeypatch):
    synthetic_summary_case(tmp_path, monkeypatch)
    summary = evaluate.summarize(tmp_path)
    assert summary['comparison_cells'] == 3456
    assert summary['new_cells'] == 2016 and summary['reused_cells'] == 1440
    assert summary['new_training_sessions'] == summary['optimizer_steps'] == 0
    assert summary['new_evaluation_frames'] == 2016
    for method in evaluate.NAMES:
        values = summary['methods'][method]
        for ep in evaluate.POINTS:
            assert values[str(ep)]['cells'] == 144
            assert values[str(ep)]['reward'] == ep + 23.5
            assert values[str(ep)]['completion_percent'] == 50
            assert values[str(ep)]['model_kib'] == 1
            assert values[str(ep)]['evicted_kib'] == .5
            assert values[str(ep)]['model_kib_per_completed_request'] == .125
        for pair, delta in [('0_to_32',32),('32_to_64',32),('0_to_64',64)]:
            assert summary['paired'][method][pair]['mean_reward_delta'] == delta
            assert summary['paired'][method][pair]['positive'] == 144
        assert values['seeds']['7']['64']['cells'] == 48
        assert summary['strata'][method]['session']['3']['64']['cells'] == 72


@pytest.mark.parametrize('field,bad', [('tensor_sha256','bad'),('actor_sha256','bad'),
                                       ('agent','ppo'),('seed',17),('episode',64)])
def test_fresh_record_provenance_mismatch_rejected(tmp_path, monkeypatch, field, bad):
    synthetic_summary_case(tmp_path, monkeypatch)
    refresh_group_rows(tmp_path, lambda rows:rows[0].__setitem__(field,bad))
    with pytest.raises(AssertionError):
        evaluate.summarize(tmp_path)


def test_fresh_frame_ledger_mismatch_rejected(tmp_path, monkeypatch):
    synthetic_summary_case(tmp_path, monkeypatch)
    refresh_group_rows(tmp_path, lambda rows:rows[0].__setitem__('steps',2))
    with pytest.raises(AssertionError):
        evaluate.summarize(tmp_path)


def test_evaluation_budget_excess_rejected(tmp_path, monkeypatch):
    contract=synthetic_summary_case(tmp_path, monkeypatch)
    contract['max_new_evaluation_frames']=2015
    with pytest.raises(AssertionError):
        evaluate.summarize(tmp_path)


def test_duplicate_group_cell_rejected(tmp_path, monkeypatch):
    synthetic_summary_case(tmp_path, monkeypatch)
    refresh_group_rows(tmp_path, lambda rows:rows.__setitem__(1,rows[0].copy()))
    with pytest.raises(AssertionError):
        evaluate.summarize(tmp_path)


def test_recorder_does_not_change_selected_control_or_rng():
    from dataclasses import dataclass
    import random
    import numpy as np
    import torch
    @dataclass
    class Control:
        execution_target: str = 'rsu_a'
    control=Control()
    class Agent:
        def act_controls(self,decisions,frame,deterministic):
            assert deterministic
            return {'v':control},None
    recorder=evaluate.DecisionRecorder(Agent(),io.StringIO(),dict(agent='test',seed=7,episode=0,case_id='dev_000'))
    py=random.getstate();np_before=np.random.get_state();tensor_before=torch.get_rng_state().clone()
    chosen,rollout=recorder.act_controls({'v':{'semantic_state':{'current_workflow_node':{'node_id':'n'}}}},0,deterministic=True)
    assert chosen['v'] is control and rollout is None and recorder.frames==1
    assert random.getstate()==py
    assert np.array_equal(np.random.get_state()[1],np_before[1])
    assert torch.equal(torch.get_rng_state(),tensor_before)


def test_zero_training_guard_raises():
    with pytest.raises(RuntimeError,match='forbids'):
        evaluate.forbidden_update([])


def test_failed_marker_prevents_restart_and_summary(tmp_path):
    (tmp_path / 'failure.json').write_text('{"status":"failed"}')
    for entry in (evaluate.run, evaluate.summarize):
        with pytest.raises(AssertionError, match='Failed evaluation'):
            entry(tmp_path)
