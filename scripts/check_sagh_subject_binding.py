"""Create-only zero-interaction subject-binding check on existing execution traces."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sagh_subject_binding import CONTRACT, bind_subject_transition, guard_prepare_context
from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs.action_schema import ActionAdapter


def forbidden(*args, **kwargs):
    raise AssertionError('Zero-interaction check: environment execution forbidden')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ledger():
    obj = object.__new__(VecWorkflowCoreEnv)
    obj._prepare_history = []
    obj._handoff_prepare_window = 6
    return obj


def check(run_root, output_root):
    paths = sorted(run_root.glob('variants/*/cells/*/seed*/steps.jsonl'))
    if not paths:
        raise ValueError('No existing step traces found')
    output_root.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
    hashes = {str(p.resolve()): digest(p) for p in paths}
    for p in [root/'scripts/sagh_subject_binding.py', Path(__file__), root/'src/envs/core/vec_workflow_core_env.py', root/'src/envs/specs/action_schema.py']:
        hashes[str(p)] = digest(p)
    write(output_root/'execution_receipt.json', dict(
        schema=CONTRACT, source_git=git, argv=sys.argv,
        started_at=datetime.now(timezone.utc).isoformat(), input_sha256=hashes,
        new_environment_steps_cap=0, learner_calls_cap=0, optimizer_steps_cap=0,
        scope='diagnostic_only_not_live_environment_integration',
        prediction_origin='legacy_per_vehicle_lookup_reconstructed_not_native_snapshot_identity',
        retain_all_legacy_exposures_rewards=True,
    ))
    for name in ('__init__', 'reset', 'step'):
        setattr(VecWorkflowCoreEnv, name, forbidden)
    ReplayProvider.step = forbidden
    adapter = ActionAdapter()
    summary = {}; register = []; exact_checks = 0; boundary_contexts = 0
    for path in paths:
        cell = str(path.relative_to(run_root).parent)
        old = bound = None
        episode_subject = None
        boundary_latched = False
        episode = -1
        for ordinal, line in enumerate(path.read_text().splitlines()):
            r = json.loads(line); s = r['semantic_state']; m = r['metadata']; q = r['metrics_protocol']
            if m['step'] == 0:
                old = ledger(); bound = ledger(); episode += 1
                episode_subject = s['primary_vehicle_id']; boundary_latched = False
            if old is None:
                raise ValueError('Trace does not start at a recorded reset')
            decision = s['primary_vehicle_id']; execution = q['cache_event']['vehicle_id']
            b = bind_subject_transition(
                episode_vehicle_id=episode_subject, decision_vehicle_id=decision,
                prediction_vehicle_id=decision,
                decision_time_index=s['time_index'],
                prediction_time_index=s['predictions'].get('prediction_time'),
                execution_time_index=q['time_index'], execution_vehicle_id=execution,
                prediction_origin='legacy_per_vehicle_lookup_reconstructed',
                decision_subject_present=any(x['vehicle_id'] == decision for x in s['vehicles']),
                execution_subject_present=bool(execution),
                boundary_already_recorded=boundary_latched,
            )
            first_boundary = not b.bound and not boundary_latched
            boundary_latched = boundary_latched or not b.bound
            action = r['actual_action']; control = adapter.decode(action, s)
            assert control.metadata['action_id'] == action  # including action 0
            kw = dict(control=control, vehicle_id=execution, required_adapter=q['required_adapter'],
                      prediction_snapshot={'predicted_handoff_target_rsu_id':q['predicted_handoff_target_rsu_id']})
            for obj in (old, bound):
                obj._episode_steps = m['step'] + 1
                obj._handoff_prepare_window = q['migration_prepare_window']
                obj._mobility_provider = SimpleNamespace(get_time=lambda: q['time_index'])
                obj._prune_prepare_history()
            context = old._build_prepare_action_context(**kw)
            guarded, reason = guard_prepare_context(context, b)
            original = old._consume_realized_prepare(execution, q['post_action_associated_rsu_id'], q['required_adapter'], q['handoff_event_count'], context)
            assert original['realized'] == q['migration_prepare_realized']
            assert original['source'] == q['migration_prepare_realized_source']
            assert original['prepare_age'] == q['migration_prepare_age']
            assert original.get('required_adapter_match',False) == q['migration_prepare_required_adapter_match']
            preview = bound._consume_realized_prepare(execution, q['post_action_associated_rsu_id'], q['required_adapter'], q['handoff_event_count'], guarded) if b.bound else None
            old._register_prepare_action(context, original)
            if b.bound:
                assert guarded == context and preview == original
                bound._register_prepare_action(guarded, preview)
                assert bound._prepare_history == old._prepare_history
                exact_checks += 1
            elif first_boundary:
                bound._prepare_history.clear()  # private preview ends at boundary; no implicit new episode
            if context is not None and guarded is None:
                boundary_contexts += 1
            comp = r['reward_breakdown']
            total = sum(comp[x] for x in ['service_reward','continuity_bonus','mechanism_exploration_bonus','positive_offset']) - sum(comp[x] for x in ['delay_penalty','cache_miss_penalty','migration_cost','constraint_penalty'])
            assert abs(total - r['native_reward']) < 1e-6
            phase_key = cell + '/' + m['phase']
            out = summary.setdefault(phase_key, dict(rows=0, episodes=0, bound=0, first_boundaries=0, after_boundary_rows=0, blocked_prepare_contexts=0, action_zero_rows=0, original_native_reward_sum=0., original_service_reward_sum=0., reasons=Counter()))
            out['rows'] += 1; out['episodes'] += m['step'] == 0; out['bound'] += b.bound
            out['first_boundaries'] += first_boundary
            out['after_boundary_rows'] += (not b.bound and not first_boundary)
            out['blocked_prepare_contexts'] += context is not None and guarded is None
            out['action_zero_rows'] += action == 0
            out['original_native_reward_sum'] += r['native_reward']
            out['original_service_reward_sum'] += comp['service_reward']
            out['reasons'][b.reason] += 1
            register.append(dict(cell=cell, ordinal=ordinal, episode=episode, phase=m['phase'], unit=m['unit'], step=m['step'], window=m['window_id'], workflow=m['workflow_id'],
                                 **b.to_dict(), first_boundary=first_boundary, actual_action=action,
                                 original_prepare_context=context, guarded_prepare_context=guarded,
                                 prepare_guard_reason=reason, original_realization=original,
                                 bound_private_preview=preview, original_native_reward=r['native_reward'],
                                 original_reward_breakdown=comp, row_retained=True))
    assert all(digest(p) == h for p,h in hashes.items())
    columns = list(register[0])
    with (output_root/'all_transition_bindings.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in register)
    result = dict(status='PASS', contract=CONTRACT, source_git=git,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  rows_retained=len(register), exact_bound_ledger_checks=exact_checks,
                  blocked_prepare_contexts=boundary_contexts, cells=summary,
                  new_environment_steps=0, learner_calls=0, optimizer_steps=0,
                  original_rewards_all_preserved=True, candidate_service_rewards_computed=False,
                  production_environment_integrated=False,
                  limitation='Existing snapshots reconstruct per-vehicle lookup origin; no native identity witness, no rollout, no gain or formal/holdout evidence.')
    write(output_root/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    check(args.run_root.resolve(), args.output_root.resolve())
