"""Synthetic consumer tests, with no policies, fitting or environment steps."""
from copy import deepcopy
import json
import pytest
from scripts.analyze_stream_learning_attribution import METHODS, SEEDS, POINTS, consume, analyze, digest


def fixture():
    cases = ['case_%02d' % i for i in range(12)]
    p = dict(methods=list(METHODS), seeds=list(SEEDS), checkpoint_episodes=list(POINTS),
             episodes_per_cell=96, training_case_ids=cases, probe_case_ids=cases[:3], source_commit='frozen')
    a = dict(status='curve_receipts_and_checkpoints_audited', source_commit='frozen',
             final_selection='episode96_fixed_no_peak', training_case_ids=cases,
             probe_case_ids=cases[:3], cells=[], rules=[])
    plot = dict(selection='fixed_final_no_peak_selection', episodes=list(POINTS), seeds=list(SEEDS), series=[], gain_rule_probe_mean=10)
    for c in cases:
        a['rules'].append(dict(case_id=c, reward=10, completed_requests=2, request_count=3,
                              steps=4, state_transfer_bytes=5, handoff_failed_count=1, model_transfer_link_frames=2))
    for mi, method in enumerate(METHODS):
        series = dict(method=method, seed_returns=[], seed_frames=[])
        for seed in SEEDS:
            rows, cps, values = [], [], []
            for ep in POINTS:
                value = 20-mi+seed/100+ep/100
                values.append(value);cps.append(dict(episode=ep, cumulative_training_frames=ep*4))
                for c in (cases if ep == 96 else cases[:3]):
                    rows.append(dict(method=method, seed=seed, episode=ep, case_id=c, reward=value,
                        completed_requests=2, request_count=3, steps=4, state_transfer_bytes=5,
                        handoff_failed_count=1, model_transfer_link_frames=2))
            a['cells'].append(dict(method=method, seed=seed, evaluation=rows, checkpoints=cps,
                                  training_frames=384, gradient_steps=24))
            series['seed_returns'].append(values);series['seed_frames'].append([e*4 for e in POINTS])
        plot['series'].append(series)
    return p, a, plot


def test_descriptive_initial_gap_is_not_claimed_as_learning():
    p, a, plot = fixture();s, seeds, gaps, finals = consume(p, a, plot)
    assert len(seeds) == 20 and len(gaps) == 15 and len(finals) == 240
    assert s['paired_gap_decomposition']['mappo']['initial_method_gap'] == pytest.approx(1)
    assert s['paired_gap_decomposition']['mappo']['relative_learning_change'] == pytest.approx(0)
    assert s['scientific_interactions'] == 0 and not s['paper_goal_complete']
    assert 'Descriptive' in s['interpretation']


@pytest.mark.parametrize('change', ['missing_seed', 'duplicate_cell', 'missing_point', 'wrong_probe',
    'plot_tamper', 'frame_tamper', 'method_tamper', 'nonfinite', 'duplicate_rule', 'peak_selection'])
def test_incomplete_or_inconsistent_receipts_rejected(change):
    p, a, plot = fixture()
    if change == 'missing_seed':a['cells'].pop()
    elif change == 'duplicate_cell':a['cells'][-1] = deepcopy(a['cells'][0])
    elif change == 'missing_point':a['cells'][0]['evaluation'].pop(0)
    elif change == 'wrong_probe':plot['series'][0]['seed_returns'][0][-1] += 7
    elif change == 'plot_tamper':plot['series'][0]['seed_returns'][0][0] += 1
    elif change == 'frame_tamper':plot['series'][0]['seed_frames'][0][-1] += 1
    elif change == 'method_tamper':a['cells'][0]['evaluation'][0]['method'] = 'ppo'
    elif change == 'nonfinite':a['cells'][0]['evaluation'][0]['reward'] = float('nan')
    elif change == 'duplicate_rule':a['rules'][-1] = deepcopy(a['rules'][0])
    elif change == 'peak_selection':a['final_selection'] = 'best_reward'
    with pytest.raises(ValueError):consume(p, a, plot)


def test_hash_binding_and_create_only_output(tmp_path):
    p, a, plot = fixture();cp=tmp_path/'campaign.json';ap=tmp_path/'audit.json';pp=tmp_path/'plot.json'
    cp.write_text(json.dumps(p));sha=digest(cp);a['campaign_sha256']=sha;ap.write_text(json.dumps(a));ash=digest(ap)
    plot.update(audit_path=str(ap),audit_sha256=ash);pp.write_text(json.dumps(plot))
    out=tmp_path/'output';result=analyze(cp,ap,pp,sha,ash,out)
    assert result['scientific_interactions'] == 0
    assert (out/'paired_gap_decomposition.csv').exists()
    with pytest.raises(FileExistsError):analyze(cp,ap,pp,sha,ash,out)
    with pytest.raises(ValueError):analyze(cp,ap,pp,'wrong',ash,tmp_path/'other')
    plot['audit_sha256']='wrong';pp.write_text(json.dumps(plot))
    with pytest.raises(ValueError):analyze(cp,ap,pp,sha,ash,tmp_path/'third')
