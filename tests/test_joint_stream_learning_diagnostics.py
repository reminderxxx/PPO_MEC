"""Synthetic qualification of diagnostic software, never scientific performance.

No archived data/checkpoints are read. No environment is constructed or stepped;
no training/update/optimizer step runs. All three checkpoint labels deliberately
refer to one initial model: this tests instrumentation, not learning effects.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_joint_stream_learning as runner
from scripts import joint_stream_same_state_scoring as score
from scripts import train_joint_stream_control as train


METHODS = ('sa_ghmappo', 'cache_offload_drl', 'dqn', 'mappo', 'dag_offload_drl')
PHASES = ('first_execution_pass', 'second_cache_event_pass')


def _forbidden(*args, **kwargs):
    raise AssertionError('Synthetic diagnostics must not execute environments or training')


@pytest.fixture(autouse=True)
def forbid_execution_and_optimizer(monkeypatch):
    monkeypatch.setattr(train, 'make_env', _forbidden)
    monkeypatch.setattr(train, 'stream_episode', _forbidden)
    monkeypatch.setattr(train.MultiRequestStreamEnv, '__init__', _forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', _forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, 'step', _forbidden)


@pytest.fixture(scope='module', params=METHODS)
def initial_agent(request):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with score.preserved_rng():
            agent = train.make_agent(request.param, 17, 64)
        yield request.param, agent
    finally:
        torch.set_num_threads(old_threads)


def _batch():
    batch = dict(
        own=torch.zeros(1, 20), critic=torch.zeros(1, 36),
        graph=torch.zeros(1, 64, 4), adjacency=torch.zeros(1, 64, 64),
        graph_mask=torch.arange(64)[None] < 2,
        prior0=torch.tensor([[0., -.2, -.1, -.3, -.4, 0.]]),
        domain=torch.zeros(1, 4), opportunity_count=torch.zeros(1, dtype=torch.long),
    )
    for head, count in ((0, 6), (1, 9), (2, 20)):
        features = torch.zeros(1, count, 32)
        mask = torch.ones(1, count, dtype=torch.bool)
        for index in range(count):
            site = index - 1 if head == 0 else ((index - 1) // 2) % 4 if head == 1 else index // 4 - 1
            if site in range(4):
                features[0, index, [0, 1, 4, 6, 16 + site]] = 1
            features[0, index, 12] = 1
            features[0, index, 15] = (index % 4) / 3
        if head == 0:
            mask[:] = torch.tensor([[False, True, True, True, True, False]])
        batch[f'f{head}'] = features
        batch[f'm{head}'] = mask
        batch[f'node{head}'] = torch.zeros(1, count, dtype=torch.long)
    return batch


def _snapshot(method, agent, phase):
    batch = _batch()
    kwargs = dict(deterministic=True)
    if phase == PHASES[0]:
        kwargs['execution_only'] = True
    else:
        kwargs.update(execution_actions=torch.tensor([2]), return_details=True)
    result, trace = score.trace_evaluate(agent, batch, kwargs)
    actions = result[0].detach().clone()
    candidates = [list(range(6)), list(range(9)), list(range(20))]
    controller = {'synthetic_context': 'same_for_every_variant'}
    return dict(
        witness=dict(agent=method, seed=17, case_id='synthetic_not_environment'),
        frame=0, phase=phase, batch=batch, reference_actions=actions,
        reference_trace=trace,
        feature=dict(vehicle_id='synthetic_vehicle', node_id='synthetic_node',
                     candidates=candidates, controller=controller),
        state_identity=dict(
            batch_sha256=score.tensor_digest(batch),
            candidates_sha256=score.json_digest(candidates),
            context_sha256=score.json_digest(controller),
            reference_prefix_sha256=score.tensor_digest({'actions': actions}),
        ),
    )


@pytest.mark.parametrize('phase', PHASES)
def test_same_state_masks_prefix_weights_and_rng_are_preserved(initial_agent, monkeypatch, phase):
    method, agent = initial_agent
    monkeypatch.setattr(agent, 'update', _forbidden)
    snapshot = _snapshot(method, agent, phase)
    before_weights = score.module_digest(agent)
    before_rng = score.rng_digest()
    before_batch = score.tensor_digest(snapshot['batch'])
    calls = []
    original_evaluate = agent._evaluate

    def observe(this, batch, **kwargs):
        calls.append((score.tensor_digest(batch), kwargs['actions'].detach().clone()))
        return original_evaluate(batch, **kwargs)

    # The labels are software controls, not independently trained checkpoints.
    models = {label: agent for label in ('0', '32', '64')}
    with score.patched_method(agent, '_evaluate', observe):
        result = score.score_snapshot(snapshot, models)
    assert len(calls) == 4
    for digest, prefix in calls:
        assert digest == before_batch
        assert torch.equal(prefix, snapshot['reference_actions'])
    assert score.module_digest(agent) == before_weights
    assert score.rng_digest() == before_rng
    assert score.tensor_digest(snapshot['batch']) == before_batch
    assert result['guards'] == dict(reference64_parity=True, actual_masks_identical=True,
                                   environment_steps=0, optimizer_steps=0)
    masks = [[head['actual_mask'] for head in variant['heads']]
             for variant in result['variants'].values()]
    assert all(mask == masks[0] for mask in masks)
    assert masks[0][0] == [False, True, True, True, True, False]
    for variant in result['variants'].values():
        assert all(variant['guards'].values())
        assert all(head['reference_is_valid'] for head in variant['heads'])
    for head in result['variants']['prior_only_64']['heads']:
        assert all(value == 0 for value in head['valid_learned_logits'])
        assert head['valid_total_logits'] == head['valid_public_prior']
    for label in ('0', '32'):
        assert result['variants'][label]['heads'] == result['variants']['64']['heads']


@pytest.mark.parametrize('raise_after_forward', (False, True))
def test_prior_zero_hooks_restore_after_success_and_exception(initial_agent, raise_after_forward):
    _, agent = initial_agent
    batch = _batch()
    kwargs = dict(deterministic=True, execution_only=True)
    _, baseline = score.trace_evaluate(agent, batch, kwargs)
    before_hooks = [dict(head._forward_hooks) for head in agent.heads]
    had_choice = '_choice_distribution' in agent.__dict__
    original_choice = agent.__dict__.get('_choice_distribution')
    before_weights, before_rng = score.module_digest(agent), score.rng_digest()
    original_evaluate = agent._evaluate

    def failing_forward(data, **options):
        original_evaluate(data, **options)
        raise RuntimeError('planned failure after hooked forward')

    if raise_after_forward:
        with pytest.raises(RuntimeError, match='planned failure'):
            score.trace_evaluate(agent, batch, kwargs, zero_learned=True, call=failing_forward)
    else:
        _, zeroed = score.trace_evaluate(agent, batch, kwargs, zero_learned=True)
        assert all(value == 0 for head in zeroed['heads'] for value in head['valid_learned_logits'])
    assert [dict(head._forward_hooks) for head in agent.heads] == before_hooks
    assert ('_choice_distribution' in agent.__dict__) == had_choice
    if had_choice:
        assert agent.__dict__['_choice_distribution'] is original_choice
    assert score.module_digest(agent) == before_weights
    assert score.rng_digest() == before_rng
    _, restored = score.trace_evaluate(agent, batch, kwargs)
    assert restored['heads'] == baseline['heads']


@pytest.mark.parametrize('raise_inside', (False, True))
def test_capture_tap_restores_instance_methods(initial_agent, raise_inside):
    _, agent = initial_agent
    names = ('act_controls', '_evaluate', '_coordinated_features')
    previous = {name: (name in agent.__dict__, agent.__dict__.get(name)) for name in names}
    before_weights, before_rng = score.module_digest(agent), score.rng_digest()
    if raise_inside:
        with pytest.raises(RuntimeError, match='planned capture failure'):
            with score.CaptureTap(agent, dict(agent='synthetic', seed=17, case_id='synthetic')):
                raise RuntimeError('planned capture failure')
    else:
        with score.CaptureTap(agent, dict(agent='synthetic', seed=17, case_id='synthetic')):
            pass
    for name, (had, value) in previous.items():
        assert (name in agent.__dict__) == had
        if had:
            assert agent.__dict__[name] is value
    assert score.module_digest(agent) == before_weights
    assert score.rng_digest() == before_rng


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_contract(tmp_path):
    root, out, source = (tmp_path / name for name in ('repo', 'output', 'source'))
    root.mkdir()
    out.mkdir()
    source.mkdir()
    paths = {}
    for label, path in (
        ('script', root / 'scripts/diagnostic.py'),
        ('science', root / 'src/science.py'),
        ('document', root / 'docs/contract.md'),
        ('asset', source / 'protocol/protocol.pt'),
        ('checkpoint', source / 'sa_ghmappo/checkpoint.pt'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('synthetic bytes: ' + label).encode())
        paths[label] = path
    witnesses = [dict(method='sa_ghmappo', seed=17, case_id=f'synthetic_{i:03d}') for i in range(15)]
    paths['plan'] = root / 'stage1/points.json'
    _write_json(paths['plan'], dict(fixed_witnesses=dict(cases=witnesses)))
    protocol = dict(source_commit='synthetic_commit', action_contract='joint_stream_control_v2',
                    source_fingerprints={'src/science.py': _sha(paths['science'])},
                    assets={'protocol.pt': _sha(paths['asset'])}, methods=['sa_ghmappo'])
    paths['protocol'] = source / 'protocol/manifest.json'
    _write_json(paths['protocol'], protocol)
    paths['dev'] = source / 'sa_ghmappo/dev.jsonl'
    paths['dev'].write_text(''.join(json.dumps(dict(witness, episode=64)) + '\n'
                                    for witness in witnesses), encoding='utf-8')
    manifest = dict(status='complete', source_commit='synthetic_commit', git_dirty=False,
                    protocol_sha256=_sha(paths['protocol']),
                    output_sha256={'dev.jsonl': _sha(paths['dev'])},
                    checkpoints=[dict(path='checkpoint.pt', seed=17, episode=64,
                                      sha256=_sha(paths['checkpoint']))])
    _write_json(source / 'sa_ghmappo/manifest.json', manifest)
    contract = dict(
        witnesses=witnesses, source_run=str(source), source_commit='synthetic_commit',
        max_reference_frames=7200, max_reference_sessions=15,
        max_frames_per_session=480, max_extra_frames=8,
        script_sha256={'scripts/diagnostic.py': _sha(paths['script'])},
        score_snapshots=[0, 32, 64], score_variants=['0', '32', '64', 'prior_only_64'],
        stage1_task_points=str(paths['plan']), stage1_task_points_sha256=_sha(paths['plan']),
        contract_document='docs/contract.md', contract_document_sha256=_sha(paths['document']),
        source_protocol_sha256=_sha(paths['protocol']),
    )
    _write_json(out / 'contract.json', contract)
    return root, out, contract, paths


def test_runner_accepts_complete_synthetic_contract(synthetic_contract):
    root, out, contract, _ = synthetic_contract
    checked, _, _, rows, verification = runner.verify_contract(root, out)
    assert checked == contract
    assert len(rows) == 15
    assert verification['science_source_checks'] == 1
    assert verification['checkpoint_count'] == 1


def test_runner_rejects_different_same_length_witness_list(synthetic_contract):
    root, out, contract, _ = synthetic_contract
    changed = deepcopy(contract)
    changed['witnesses'][0]['case_id'] = 'synthetic_unapproved_case'
    _write_json(out / 'contract.json', changed)
    with pytest.raises(ValueError, match='Witness list differs'):
        runner.verify_contract(root, out)


@pytest.mark.parametrize('field', (
    'max_reference_frames', 'max_reference_sessions', 'max_frames_per_session', 'max_extra_frames',
))
def test_runner_rejects_budget_expansion(synthetic_contract, field):
    root, out, contract, _ = synthetic_contract
    contract[field] += 1
    _write_json(out / 'contract.json', contract)
    with pytest.raises(ValueError, match='Unapproved stage2 cap'):
        runner.verify_contract(root, out)


@pytest.mark.parametrize('label, message', (
    ('script', 'script hash mismatch'),
    ('plan', 'fixed witness plan changed'),
    ('document', 'document changed'),
    ('science', 'science source changed'),
    ('asset', 'protocol asset changed'),
    ('checkpoint', 'checkpoint changed'),
    ('dev', 'dev changed'),
    ('protocol', 'protocol hash mismatch'),
))
def test_runner_rejects_modified_frozen_bytes(synthetic_contract, label, message):
    root, out, _, paths = synthetic_contract
    # Appended whitespace keeps JSON valid while changing its byte hash.
    with paths[label].open('ab') as stream:
        stream.write(b' \n')
    with pytest.raises(ValueError, match=message):
        runner.verify_contract(root, out)


@pytest.mark.parametrize('field, value', (
    ('score_snapshots', [0, 16, 32, 64]),
    ('score_variants', ['0', '32', '64', 'prior_only_64', 'extra_policy']),
))
def test_runner_rejects_expanded_scoring_set(synthetic_contract, field, value):
    root, out, contract, _ = synthetic_contract
    contract[field] = value
    _write_json(out / 'contract.json', contract)
    with pytest.raises(ValueError, match='Unapproved scoring variants'):
        runner.verify_contract(root, out)
