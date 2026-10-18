"""Bounded throwaway SA gradient-clip software qualification; no scientific training.

Consumes only three already-frozen stage2 dev_046 snapshot packs. A PPO-form
loss on greedy reference actions is a software witness, not an original training
batch, off-policy estimate, resumed training, performance claim or causal proof.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_joint_stream_cached_gradients import load_scorer, object_digest, pack_rows
from scripts.train_joint_stream_control import NAMES, actor_hash, make_agent
from scripts.train_joint_tail_control import tensor_hash
from src.agents.ppo_agent import (CoordinatedControlPPO, DeliveryControlPPO,
    ParameterizedControlPPO, ScheduledControlPPO, StreamControlPPO, TailControlPPO)
from src.agents.sa_ghmappo_agent import StreamSAGHMAPPOAgent

REFERENCE = '35975cb3d73c84ddbabf01f6749bbc2d60c826de'
SEEDS = (7, 17, 27)
POINTS = ('0', '32', '64')
VALUE_PREFIXES = ('critic.', 'prefix_values.', 'action_values.')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def optimizer_pair(model):
    return {'main': model.optimizer.state_dict(), 'surrogate': model.surrogate_optimizer.state_dict()}


def full_model_digest(model):
    return object_digest({'state': model.state_dict(), 'optimizers': optimizer_pair(model),
                          'grads': {n: p.grad for n, p in model.named_parameters()}})


def policy_names(model):
    return [n for n, p in model.named_parameters()
            if not n.startswith(('surrogates.',) + VALUE_PREFIXES)]


def named_subset(model, names, *, gradients=False):
    params = dict(model.named_parameters())
    return {n: (None if params[n].grad is None else params[n].grad.detach().clone())
            if gradients else params[n].detach().clone() for n in names}


def policy_adam_state(model, names):
    params = dict(model.named_parameters())
    return {n: deepcopy(model.optimizer.state.get(params[n], {})) for n in names}


def clone_model(reference, *, kind):
    options = deepcopy(reference.options)
    options.pop('separate_policy_value_clip', None)
    if kind == 'parent':
        result = StreamControlPPO(**options)
    else:
        result = StreamSAGHMAPPOAgent(separate_policy_value_clip=kind in ('candidate_a', 'candidate_b'), **options)
    result.load_state_dict(deepcopy(reference.state_dict()))
    result.optimizer.load_state_dict(deepcopy(reference.optimizer.state_dict()))
    result.surrogate_optimizer.load_state_dict(deepcopy(reference.surrogate_optimizer.state_dict()))
    require(state_equal(result.state_dict(), reference.state_dict()), 'Clone parameter identity failed')
    require(state_equal(optimizer_pair(result), optimizer_pair(reference)), 'Clone Adam identity failed')
    return result


class QualificationGuard:
    """Allow only registered throwaway Adam optimizers; forbid environments/updates."""
    def __init__(self, limit):
        self.limit = limit
        self.steps = []
        self.allowed = {}
        # Keep registered optimizers alive so Python cannot recycle their IDs.
        self.optimizer_keepalive = []
        self.blocked = []
        self.patches = []
        self.apply_calls = 0

    def register(self, model, identity):
        for label, optimizer in [('main', model.optimizer), ('surrogate', model.surrogate_optimizer)]:
            require(type(optimizer) is torch.optim.Adam, 'Only Adam is approved')
            require(id(optimizer) not in self.allowed, 'Optimizer registered twice')
            self.allowed[id(optimizer)] = dict(identity, optimizer=label)
            self.optimizer_keepalive.append(optimizer)

    def patch(self, obj, name, value):
        had = name in obj.__dict__
        previous = obj.__dict__.get(name)
        self.patches.append((obj, name, had, previous))
        setattr(obj, name, value)

    def forbidden(self, kind):
        def stop(*args, **kwargs):
            self.blocked.append(kind)
            raise RuntimeError('Qualification forbids ' + kind)
        return stop

    def __enter__(self):
        original_adam_step = torch.optim.Adam.step
        guard = self
        def counted_step(optimizer, *args, **kwargs):
            require(id(optimizer) in guard.allowed, 'Adam.step on unregistered/original optimizer forbidden')
            require(len(guard.steps) < guard.limit, 'Qualification Adam.step budget exceeded')
            result = original_adam_step(optimizer, *args, **kwargs)
            guard.steps.append(dict(guard.allowed[id(optimizer)], ordinal=len(guard.steps) + 1))
            return result
        self.patch(torch.optim.Adam, 'step', counted_step)
        self.patch(torch.optim.AdamW, 'step', self.forbidden('AdamW.step'))
        self.patch(torch.optim.SGD, 'step', self.forbidden('SGD.step'))
        for cls in (ParameterizedControlPPO, CoordinatedControlPPO, ScheduledControlPPO,
                    TailControlPPO, DeliveryControlPPO, StreamControlPPO, StreamSAGHMAPPOAgent):
            self.patch(cls, 'update', self.forbidden('full update()'))
        self.patch(StreamControlPPO, 'act_controls', self.forbidden('act_controls/rollout'))
        self.patch(StreamControlPPO, '_coordinated_features', self.forbidden('live feature/environment decision'))
        from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
        for name in ('__init__', 'reset', 'step_controls'):
            self.patch(MultiRequestStreamEnv, name, self.forbidden('environment.' + name))
        import scripts.train_joint_stream_control as training_module
        self.patch(training_module, 'make_env', self.forbidden('make_env'))
        return self

    def __exit__(self, typ, value, tb):
        for obj, name, had, previous in reversed(self.patches):
            if had:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        return False


def ppo_form_loss(model, data, labels, *, value_scale):
    actions = labels['actions']
    active = labels['active_heads']
    target = labels['team_credit_targets']
    advantage = target[:, None] - labels['counterfactual_baselines']
    for j in range(3):
        sample = advantage[active[:, j], j]
        if len(sample) > 1:
            advantage[:, j] = (advantage[:, j] - sample.mean()) / sample.std().clamp(min=1e-6)
    outputs = model._evaluate(data, actions=actions, deterministic=True, return_details=True)
    _, logs, entropy, value, surrogate, prefix, enabled, baseline, q, _ = outputs
    require(torch.equal(enabled, active), 'Reference active-mask mismatch')
    ratio = (logs - labels['head_logs']).exp()
    weight = enabled.float()
    denominator = weight.sum().clamp(min=1)
    objective = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
    policy = -(objective * weight).sum() / denominator
    value_loss = (prefix - target[:, None]).square().mean() + (value - target).square().mean() + (q - target[:, None]).square().mean()
    expected = torch.cat((target[:, None], labels['cost_targets']), -1)[:, None]
    auxiliary = (surrogate - expected).square().mean() if model.surrogate_enabled else value_loss * 0
    progress = min(1., float(model.completed_updates) * 4 / max(1, model.training_budget - 4))
    entropy_loss = -model._entropy_coefficient(progress) * (entropy * weight).sum() / denominator
    terms = {'policy': policy, 'value_half': .5 * value_loss,
             'entropy': entropy_loss, 'surrogate_point2': .2 * auxiliary}
    total = terms['policy'] + value_scale * terms['value_half'] + terms['entropy'] + terms['surrogate_point2']
    require(bool(torch.isfinite(total)), 'Nonfinite PPO-form loss')
    names = policy_names(model)
    mapping = dict(model.named_parameters())
    gradients = torch.autograd.grad(total, [mapping[n] for n in names], retain_graph=True, allow_unused=True)
    raw_policy = {n: None if g is None else g.detach().clone() for n, g in zip(names, gradients)}
    return total, outputs, terms, raw_policy


def frozen_sources(root, stage_contract, source_run):
    plan = json.loads((source_run / 'protocol/manifest.json').read_text())
    require(sha(source_run / 'protocol/manifest.json') == stage_contract['source_protocol_sha256'], 'Source protocol SHA mismatch')
    require(plan['source_commit'] == stage_contract['algorithm_reference'] == REFERENCE, 'Algorithm reference mismatch')
    expected = stage_contract['original_source_fingerprints']
    require(len(expected) == 123 and expected == plan['source_fingerprints'], 'Original source fingerprints mismatch')
    allowed = stage_contract['allowed_algorithm_changes']
    require(allowed == ['src/agents/sa_ghmappo_agent.py'], 'Unapproved allowed-source scope')
    changed = []
    evidence = []
    for name, digest in expected.items():
        original = subprocess.check_output(['git', 'show', REFERENCE + ':' + name], cwd=root)
        require(hashlib.sha256(original).hexdigest() == digest, 'Original Git source mismatch: ' + name)
        live = sha(root / name)
        if live != digest:
            require(name in allowed, 'Unapproved live algorithm change: ' + name)
            changed.append(name)
        evidence.append(dict(path=name, original_git_sha256=digest, live_sha256=live))
    for asset, digest in plan['assets'].items():
        require(sha(source_run / 'protocol' / asset) == digest, 'Protocol asset mismatch')
    require('separate_policy_value_clip' in inspect.signature(StreamSAGHMAPPOAgent.__init__).parameters,
            'Install the candidate before running qualification')
    return plan, dict(original_git_source_files=len(expected), changed_live_files=changed, source_files=evidence)


def checkpoint_compatibility(source_run, plan):
    evidence = []
    for name in NAMES:
        manifest = json.loads((source_run / name / 'manifest.json').read_text())
        require(manifest['status'] == 'complete' and manifest['source_commit'] == REFERENCE, 'Unfrozen original method')
        require(manifest['protocol_sha256'] == sha(source_run / 'protocol/manifest.json'), 'Method protocol mismatch')
        require(len(manifest['checkpoints']) == 6, 'Expected six original checkpoints')
        require({(c['seed'], c['episode']) for c in manifest['checkpoints']} == {(s, e) for s in SEEDS for e in (32, 64)}, 'Checkpoint coverage mismatch')
        for item in manifest['checkpoints']:
            path = source_run / name / item['path']
            require(sha(path) == item['sha256'], 'Original checkpoint file changed')
            model = type(make_agent(name, item['seed'], plan['episodes'])).from_checkpoint(path)
            require(tensor_hash(model) == item['trained_tensor_sha256'], 'Old checkpoint tensor compatibility failed')
            require(actor_hash(model) == item['trained_actor_sha256'], 'Old checkpoint actor compatibility failed')
            if name in ('sa_ghmappo', 'sa_no_prediction'):
                require(model.separate_policy_value_clip is False, 'Old SA checkpoints must default to False')
            evidence.append(dict(agent=name, seed=item['seed'], episode=item['episode'], path=str(path),
                file_sha256=item['sha256'], tensor_sha256=tensor_hash(model), actor_sha256=actor_hash(model)))
    require(len(evidence) == 48, 'Expected48 old checkpoint compatibility checks')
    return evidence


def captured_inputs(root, stage2_root, scorer):
    completed = json.loads((stage2_root / 'result.json').read_text())
    require(completed['status'] == 'complete' and completed['reference_sessions'] == 15, 'Stage2 reference incomplete')
    require(completed['training_sessions'] == completed['optimizer_steps'] == 0, 'Stage2 reference is not zero-training')
    require(sha(stage2_root / 'contract.json') == completed['contract_sha256'], 'Stage2 contract SHA mismatch')
    packs = {}
    for seed in SEEDS:
        entries = [e for e in completed['captures'] if e['witness']['method'] == 'sa_ghmappo'
                   and int(e['witness']['seed']) == seed and e['witness']['case_id'] == 'dev_046']
        require(len(entries) == 1, 'Expected one original dev046 capture per SA seed')
        entry = entries[0]
        path = Path(entry['path'])
        if not path.is_absolute():
            path = stage2_root / path
        path = path.resolve()
        require(stage2_root.resolve() in path.parents and sha(path) == entry['sha256'], 'Capture path/SHA mismatch')
        capture = torch.load(path, map_location='cpu', weights_only=False)
        require(capture['schema'] == 'same_state_reference_capture_v1', 'Unsupported snapshot schema')
        require(capture['witness']['method'] == 'sa_ghmappo' and int(capture['witness']['seed']) == seed
                and capture['witness']['case_id'] == 'dev_046', 'Capture witness identity mismatch')
        rows, data, labels, identities = pack_rows(capture['snapshots'], scorer)
        require(len(rows) == 72, 'Qualification must use all72 frozen second-pass rows')
        require(len({(x['frame'], x['feature']['vehicle_id']) for x in rows}) == 72, 'Duplicate captured row')
        require(len(capture['selected_frames']) == 9 and all(x['frame'] in capture['selected_frames'] for x in rows), 'Frozen frame coverage mismatch')
        packs[seed] = dict(path=path, file_sha256=entry['sha256'], data=data, labels=labels,
                           identities=identities, input_sha256=object_digest((data, labels)))
    return packs


def qualify_state(reference, seed, point, packet, guard):
    data, labels = packet['data'], packet['labels']
    input_before = object_digest((data, labels))
    models = {kind: clone_model(reference, kind=kind)
              for kind in ('parent', 'false', 'candidate_a', 'candidate_b')}
    original_ids = {id(p) for p in reference.parameters()}
    require(all(not original_ids.intersection(id(p) for p in model.parameters()) for model in models.values()),
            'Qualification clone aliases original model')
    for kind, model in models.items():
        guard.register(model, dict(seed=seed, snapshot=point, kind=kind))
    states_before = {kind: full_model_digest(model) for kind, model in models.items()}
    calculations = {kind: ppo_form_loss(model, data, labels, value_scale=100. if kind == 'candidate_b' else 1.)
                    for kind, model in models.items()}
    forwards = {kind: result[1] for kind, result in calculations.items()}
    require(all(state_equal(forwards['false'], output) for output in forwards.values()), 'Candidate/default/parent forward differs before optimization')
    require(all(states_before[kind] == full_model_digest(model) for kind, model in models.items()), 'Forward/autograd.grad changed model/Adam/grad buffers')
    for name in ('policy', 'entropy', 'surrogate_point2', 'value_half'):
        require(torch.equal(calculations['candidate_a'][2][name], calculations['candidate_b'][2][name]),
                'A/B base loss component differs: ' + name)
    require(state_equal(calculations['candidate_a'][3], calculations['candidate_b'][3]),
            'Value scaling changed raw policy gradients before clipping')
    step_before = len(guard.steps)
    for kind, model in models.items():
        model._apply_control_gradients(calculations[kind][0])
        guard.apply_calls += 1
    require(len(guard.steps) - step_before == 8, 'Expected exactly eight throwaway Adam steps per state')
    parent, default = models['parent'], models['false']
    require(state_equal(parent.state_dict(), default.state_dict()), 'False does not exactly match parent parameters')
    require(state_equal(optimizer_pair(parent), optimizer_pair(default)), 'False does not exactly match parent Adam state')
    require(state_equal({n: p.grad for n, p in parent.named_parameters()},
                        {n: p.grad for n, p in default.named_parameters()}), 'False/parent clipped gradient mismatch')
    a, b = models['candidate_a'], models['candidate_b']
    names = policy_names(a)
    require(state_equal(named_subset(a, names, gradients=True), named_subset(b, names, gradients=True)),
            'Value scaling changed clipped policy gradients')
    require(state_equal(named_subset(a, names), named_subset(b, names)), 'Value scaling changed policy parameters')
    require(state_equal(policy_adam_state(a, names), policy_adam_state(b, names)), 'Value scaling changed policy Adam state')
    require(object_digest((data, labels)) == input_before, 'Qualification changed cached inputs')
    for model in models.values():
        require(all(bool(torch.isfinite(x).all()) for x in model.state_dict().values()), 'Nonfinite private model state')
        require(int(model.completed_updates) == int(reference.completed_updates), 'Software step changed scientific update counter')
    stats_a = a._separate_gradient_stats[-1]
    stats_b = b._separate_gradient_stats[-1]
    require(stats_a[0] == stats_b[0] and stats_a[2] == stats_b[2], 'Policy/surrogate preclip norms differ under value-only scaling')
    return dict(seed=seed, snapshot=point, input_rows=72, input_sha256=input_before,
        source_tensor_sha256=tensor_hash(reference), source_actor_sha256=actor_hash(reference),
        forward_exact=True, default_parent_parameters_exact=True, default_parent_adam_exact=True,
        default_parent_clipped_gradients_exact=True, candidate_value_scale_a=1., candidate_value_scale_b=100.,
        candidate_policy_raw_gradients_exact=True, candidate_policy_clipped_gradients_exact=True,
        candidate_policy_parameters_exact=True, candidate_policy_adam_exact=True,
        candidate_policy_gradient_sha256=object_digest(calculations['candidate_a'][3]),
        candidate_policy_parameter_sha256=object_digest(named_subset(a, names)),
        candidate_policy_adam_sha256=object_digest(policy_adam_state(a, names)),
        parent_global_clip_norms=list(parent._gradient_stats[-1]),
        default_global_clip_norms=list(default._gradient_stats[-1]),
        candidate_a_clip_stats=list(stats_a), candidate_b_clip_stats=list(stats_b),
        candidate_clip_stat_fields=['policy_norm', 'value_norm', 'surrogate_norm', 'policy_coefficient', 'value_coefficient', 'surrogate_coefficient'],
        ppo_form_losses={kind: {k: float(v.detach()) for k, v in item[2].items()} for kind, item in calculations.items()},
        throwaway_adam_steps=8, throwaway_apply_calls=4, environment_steps=0, full_update_calls=0,
        input_unchanged=True, old_update_counter_unchanged=True)


def run(args):
    root = args.root.resolve()
    stage3 = args.stage3_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.set_num_threads(1)
    contract_path = stage3 / 'contract.json'
    contract = json.loads(contract_path.read_text())
    record = dict(status='running', started_at=now(), stage='3-software-qualification',
        command=sys.argv, contract_path=str(contract_path), contract_sha256=sha(contract_path),
        qualifier_sha256=sha(__file__), source_algorithm_reference=REFERENCE,
        scientific_training_sessions=0, scientific_optimizer_steps=0,
        limits=['cached greedy-reference PPO-form one-step software witness only',
                'not resumed training; old checkpoints are evaluation snapshots',
                'not actual original training gradients, performance evaluation, causal learning evidence or convergence proof'])
    write_json(output / 'command.json', dict(command=sys.argv, started_at=record['started_at']))
    write_json(output / 'run_started.json', dict(started_at=record['started_at'], contract_sha256=record['contract_sha256']))
    guard = None
    try:
        require(contract['schema'] == 'sa_learning_stage3_v1', 'Unsupported stage3 contract')
        require(contract['new_scientific_training_sessions'] == contract['new_scientific_optimizer_steps'] == 0, 'Only software qualification authorized')
        require(contract['candidate'] == 'separate_policy_value_clip=True'
                and contract['candidate_training_gradient_contract'] == 'sa_policy_value_clip_v1', 'Unexpected candidate')
        budget = int(contract['max_qualification_adam_steps'])
        require(72 <= budget <= 128, 'Unapproved throwaway Adam budget')
        require(sha(root / contract['contract_document']) == contract['contract_document_sha256'], 'Contract document changed')
        scorer_path = root / 'scripts/joint_stream_same_state_scoring.py'
        scorer = load_scorer(scorer_path)
        rng_before = scorer.rng_digest()
        source_run = Path(contract['source_run']).resolve()
        with scorer.preserved_rng(), QualificationGuard(budget) as guard:
            plan, source_checks = frozen_sources(root, contract, source_run)
            checkpoint_checks = checkpoint_compatibility(source_run, plan)
            stage2_root = args.stage2_root.resolve() if args.stage2_root else root / 'artifacts/joint_stream_learning/learning_diagnosis_stage2_20261018'
            packets = captured_inputs(root, stage2_root, scorer)
            record['source_checks'] = source_checks
            record['checkpoint_checks'] = checkpoint_checks
            record['inputs'] = dict(scorer_path=str(scorer_path), scorer_sha256=sha(scorer_path),
                packer_path=str(root/'scripts/audit_joint_stream_cached_gradients.py'),
                packer_sha256=sha(root/'scripts/audit_joint_stream_cached_gradients.py'),
                stage2_result_sha256=sha(stage2_root/'result.json'), stage2_contract_sha256=sha(stage2_root/'contract.json'),
                captures=[dict(seed=seed, path=str(packets[seed]['path']), file_sha256=packets[seed]['file_sha256'],
                    packed_input_sha256=packets[seed]['input_sha256'], second_pass_rows=72) for seed in SEEDS])
            record['results'] = []
            for seed in SEEDS:
                models, provenance = scorer.load_exact_models(root, source_run, 'sa_ghmappo', seed)
                before = {point: full_model_digest(model) for point, model in models.items()}
                for point in POINTS:
                    with torch.enable_grad():
                        value = qualify_state(models[point], seed, point, packets[seed], guard)
                    value['model_provenance'] = provenance[point]
                    record['results'].append(value)
                    write_json(output/'progress.json', dict(status='running', updated_at=now(),
                        completed_states=len(record['results']), total_states=9, throwaway_adam_steps=len(guard.steps),
                        scientific_training_sessions=0, environment_steps=0))
                require(all(before[k] == full_model_digest(v) for k, v in models.items()), 'Original loaded reference models were modified')
            require(len(record['results']) == 9 and guard.apply_calls == 36 and len(guard.steps) == 72, 'Unexpected qualification operation counts')
            require(not guard.blocked, 'Forbidden operation was attempted')
            require(all(sha(row['path']) == row['file_sha256'] for row in checkpoint_checks), 'Original checkpoint file changed during qualification')
            require(all(sha(packets[s]['path']) == packets[s]['file_sha256'] for s in SEEDS), 'Original capture file changed during qualification')
            require(all(sha(root / item['path']) == item['live_sha256'] for item in source_checks['source_files']), 'Live algorithm source changed during qualification')
            record['adam_steps'] = guard.steps
        require(scorer.rng_digest() == rng_before, 'Caller RNG changed after qualification')
        require(sha(contract_path) == record['contract_sha256'], 'Stage3 contract changed during qualification')
        require(sha(__file__) == record['qualifier_sha256'], 'Qualification script changed during execution')
        record.update(status='pass', completed_at=now(),
            counts=dict(original_checkpoint_files_verified=48, original_checkpoint_compatibility_loads=48,
                additional_exact_SA_checkpoint_loads=6, reconstructed_initial_SA_models=3,
                reference_snapshot_packs=3, second_pass_rows_per_seed=72, states=9,
                private_model_copies=36, private_apply_control_gradient_calls=36,
                throwaway_adam_steps=72, throwaway_adam_step_limit=budget,
                new_scientific_training_sessions=0, new_scientific_optimizer_steps=0,
                full_update_calls=0, environment_constructions=0, environment_steps=0, new_policy_rollouts=0),
            guards=dict(caller_rng_preserved=True, cached_inputs_unchanged=True, original_loaded_models_unchanged=True,
                        all_original_checkpoint_and_capture_files_unchanged=True, only_registered_private_adam_steps=True,
                        default_equals_parent_exact=True, value_scale_policy_isolation_exact=True))
        write_json(output/'qualification.json', record)
        write_json(output/'result.json', dict(status='pass', completed_at=record['completed_at'],
            qualification_sha256=sha(output/'qualification.json'), counts=record['counts']))
        print(json.dumps(dict(status='pass', output_root=str(output), counts=record['counts'])), flush=True)
    except BaseException:
        record.update(status='failed', failed_at=now(), traceback=traceback.format_exc(),
                      throwaway_adam_steps=0 if guard is None else len(guard.steps),
                      blocked_operations=[] if guard is None else guard.blocked)
        write_json(output/'failure.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', '--output-root', type=Path, required=True)
    parser.add_argument('--stage3_root', '--stage3-root', type=Path, required=True)
    parser.add_argument('--stage2_root', '--stage2-root', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
