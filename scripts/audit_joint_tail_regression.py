"""Read-only component interventions and tail-credit diagnosis on frozen models."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import arguments, _load_protocol
from scripts.train_joint_parameterized_control import episode
from src.agents.registry import build_agent
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor
from src.envs.core.multi_request_replay_env import ReplayCosts
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv


def tensor_hash(agent):
    return hashlib.sha256(b''.join(v.numpy().tobytes() for v in agent.state_dict().values())).hexdigest()


def main():
    torch.set_num_threads(1)
    root = ROOT/'artifacts/joint_scheduled_control'
    path = next((root/'main/seed27').glob('*/manifest.json'))
    manifest = json.loads(path.read_text())
    assert manifest['status'] == 'complete'
    source = json.loads((ROOT/'artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json').read_text())
    splits, workflows, rsus = _load_protocol(arguments(source['command'][1:]))
    provider = LearnedMobilityPredictor(torch.load(path.parent/'predictor.pt', weights_only=False))

    def load(name, ep):
        p = path.parent/f'{name}_seed27_ep{ep}.pt'
        saved = torch.load(p, weights_only=False)
        agent = build_agent(saved['options']['method'], control_contract=manifest['action_contract'],
                            scheduled_controls=True, **{k:v for k,v in saved['options'].items() if k!='method'})
        return type(agent).from_checkpoint(p)

    old = load('sa_ghmappo', 96).state_dict()
    treatments = [('full96', 'sa_ghmappo', 96, None), ('full128', 'sa_ghmappo', 128, None),
                  ('no_prediction128', 'sa_no_prediction', 128, None),
                  ('full128_no_surrogate', 'sa_ghmappo', 128, 'disable_surrogate'),
                  ('full128_execution96', 'sa_ghmappo', 128, ('heads.0.',)),
                  ('full128_priority96', 'sa_ghmappo', 128, ('heads.2.',)),
                  ('full128_graph96', 'sa_ghmappo', 128, ('graph_encoder.', 'graph_message.'))]
    output = ROOT/'artifacts/joint_tail_regression_20261015'
    if '--heads' in sys.argv:
        treatments=[(f'full128_no_surrogate_head{j}','sa_ghmappo',128,j) for j in range(3)]
        output=output/'surrogate_heads'
    output.mkdir(exist_ok=True)
    rows = []
    with (output/'events.jsonl').open('w') as events:
        for label, name, ep, treatment in treatments:
            agent = load(name, ep)
            if type(treatment) is int:
                # Declared frozen-component intervention, no optimizer or checkpoint writes.
                with torch.no_grad():
                    agent.surrogates[treatment][-1].weight[0].zero_()
                    agent.surrogates[treatment][-1].bias[0].zero_()
            elif treatment == 'disable_surrogate':
                agent.surrogate_enabled = False
            elif treatment:
                state = agent.state_dict()
                state.update({k:v for k,v in old.items() if k.startswith(treatment)})
                agent.load_state_dict(state)
            before = tensor_hash(agent)
            gaps = []
            original = agent._choice_distribution
            def choice(logits, mask, deterministic):
                dist = original(logits, mask, deterministic)
                if int(mask.sum()) > 1:
                    scores = logits[mask].sort(descending=True).values
                    gaps.append(float(scores[0]-scores[1]))
                return dist
            agent._choice_distribution = choice
            for window in splits['dev']:
                gaps.clear()
                env = MultiRequestTargetControlEnv(frames=window['frames'],
                    vehicle_workflows=dict(zip(window['vehicle_ids'], workflows)), rsu_states=rsus,
                    costs=ReplayCosts(**manifest['costs']['replay_costs']),
                    prediction_provider=provider, **manifest['environment_options'])
                row, rolls = episode(env, agent, deterministic=True, event_file=events,
                    identity=dict(treatment=label, frame_offset=window['frame_offset']))
                rewards = [(e['frame_index'], sum(e['frame_cost_receipt']['reward_components'].values()))
                           for e in env.events if 'frame_cost_receipt' in e]
                tail = [e for e in env.events if e['frame_index'] >= 90]
                tail_credit = []
                by_event = {(e['frame_index'],e['vehicle_id']):e for e in env.events}
                for roll in rolls:
                    frame = round(float(roll['data']['own'][0,0])*120)
                    for i, vid in enumerate(roll['vehicle_ids']):
                        if frame < 90:
                            continue
                        event = by_event[(frame,vid)]
                        tail_credit.append(dict(frame=frame, vehicle=vid, status=event['status'],
                            credit=float(roll['credit_targets'][i]),
                            baselines=roll['counterfactual_baselines'][i].tolist(),
                            active_heads=roll['active_heads'][i].tolist()))
                rows.append(dict(treatment=label, frame_offset=window['frame_offset'],
                    reward=row['reward'], completed=row['completed_requests'], nodes=row['completed_nodes'],
                    discounted_reward=sum(.99**f*r for f,r in rewards),
                    tail_statuses=dict(Counter(e['status'] for e in tail)),
                    requests=row['per_request'], queue=row['status_counts'],
                    gap_median=float(np.median(gaps)) if gaps else None,
                    gap_below_01=sum(x<.01 for x in gaps), active_distributions=len(gaps),
                    tail_credit=tail_credit))
                print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('tail_credit','requests','queue','tail_statuses')}),flush=True)
            assert tensor_hash(agent) == before
    # Geometry only: no policy score is used to select windows or a layout.
    coverage = {}
    from math import hypot
    for split in ('train','dev'):
        windows = []
        for w in splits[split]:
            exits=changes=0
            previous = {}
            for f in w['frames']:
                for v in f['vehicles']:
                    covered={r.rsu_id for r in rsus if hypot(v.position_x-r.position_x,v.position_y-r.position_y)<=r.coverage_radius}
                    exits += len(previous.get(v.vehicle_id,covered)-covered)
                    changes += int(v.vehicle_id in previous and previous[v.vehicle_id] != covered)
                    previous[v.vehicle_id]=covered
            windows.append(dict(frame_offset=w['frame_offset'], coverage_departures=exits, changed_vehicle_frames=changes))
        coverage[split]=windows
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(), literature_cutoff='2026-10-15',
        target_venue='IEEE TMC', policy_version='tmc_review_policy_v3_20260621',
        artifact_run_id=manifest['run_id'], source_commit=manifest['git_commit'],
        diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evidence_level='frozen development interventions; paper E0_UNAVAILABLE',
        rows=rows, coverage=coverage,
        discount_witness=dict(early_one_node=1.,late_two_nodes=.99**119*2,
                             undiscounted_early=1.,undiscounted_late=2.),
        limits='component swaps diagnose frozen seed27 only; no training, selection or holdout; not independent clusters')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
