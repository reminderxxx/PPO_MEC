"""Describe initial method gaps and later learning from an immutable audited curve.

This consumer performs no policy load, environment interaction or fitting. The
gap identity is descriptive; it is not a randomized causal decomposition.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ('main', 'mappo', 'ppo', 'dqn')
SEEDS = (7, 17, 27, 37, 47)
POINTS = tuple(range(0, 97, 8))
METRICS = ('reward', 'completed_requests', 'request_count', 'steps',
           'state_transfer_bytes', 'handoff_failed_count', 'model_transfer_link_frames')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('metric must be a finite number')
    return float(value)


def near(left, right):
    if not math.isclose(finite(left), finite(right), rel_tol=0, abs_tol=1e-8):
        raise ValueError('bound curve point differs from audited evaluation')


def consume(campaign, audit, plot):
    if (campaign.get('methods') != list(METHODS) or campaign.get('seeds') != list(SEEDS)
            or campaign.get('checkpoint_episodes') != list(POINTS)
            or campaign.get('episodes_per_cell') != 96):
        raise ValueError('unexpected frozen method/seed/checkpoint design')
    if (audit.get('status') != 'curve_receipts_and_checkpoints_audited'
            or audit.get('source_commit') != campaign.get('source_commit')
            or audit.get('final_selection') != 'episode96_fixed_no_peak'
            or plot.get('selection') != 'fixed_final_no_peak_selection'
            or plot.get('episodes') != list(POINTS) or plot.get('seeds') != list(SEEDS)):
        raise ValueError('missing audited status, source or fixed-final identity')
    cases = campaign.get('training_case_ids', [])
    probes = campaign.get('probe_case_ids', [])
    if (len(cases) != 12 or len(set(cases)) != 12 or len(probes) != 3
            or len(set(probes)) != 3 or not set(probes) <= set(cases)
            or audit.get('training_case_ids') != cases or audit.get('probe_case_ids') != probes):
        raise ValueError('fixed input/probe pool differs')
    cells = {(x['method'], x['seed']): x for x in audit.get('cells', [])}
    if len(audit.get('cells', [])) != 20 or set(cells) != {(m, s) for m in METHODS for s in SEEDS}:
        raise ValueError('incomplete or duplicate method/seed cells')
    series = {x['method']: x for x in plot.get('series', [])}
    if len(plot.get('series', [])) != 4 or set(series) != set(METHODS):
        raise ValueError('incomplete or duplicate plotted methods')
    rule_rows = audit.get('rules', [])
    rules = {r['case_id']: r for r in rule_rows}
    if len(rule_rows) != 12 or set(rules) != set(cases):
        raise ValueError('rule must occur exactly once per declared condition')
    near(plot.get('gain_rule_probe_mean'), statistics.mean(finite(rules[c]['reward']) for c in probes))
    seed_rows, final_rows, endpoints = [], [], {}
    expected_eval = [(ep, c) for ep in POINTS for c in (cases if ep == 96 else probes)]
    for method in METHODS:
        s = series[method]
        if len(s.get('seed_returns', [])) != 5 or len(s.get('seed_frames', [])) != 5:
            raise ValueError('all five plotted seeds required')
        for seed_index, seed in enumerate(SEEDS):
            cell = cells[method, seed]
            rows = cell.get('evaluation', [])
            if [(r['episode'], r['case_id']) for r in rows] != expected_eval:
                raise ValueError('evaluation pool/ordering or endpoint is incomplete')
            cps = cell.get('checkpoints', [])
            if [r['episode'] for r in cps] != list(POINTS):
                raise ValueError('checkpoint sequence incomplete')
            returns, frames = s['seed_returns'][seed_index], s['seed_frames'][seed_index]
            if len(returns) != 13 or len(frames) != 13:
                raise ValueError('all thirteen points required')
            values = {}
            for i, ep in enumerate(POINTS):
                probe = [r for r in rows if r['episode'] == ep and r['case_id'] in probes]
                if len(probe) != 3:
                    raise ValueError('probe count differs')
                for r in rows:
                    if r['method'] != method or r['seed'] != seed:
                        raise ValueError('evaluation method/seed differs')
                values[ep] = statistics.mean(finite(r['reward']) for r in probe)
                near(returns[i], values[ep])
                near(frames[i], cps[i]['cumulative_training_frames'])
            endpoints[method, seed] = values[0], values[96]
            seed_rows.append(dict(method=method, seed=seed, initial_probe_return=values[0],
                                  final_probe_return=values[96], learning_change=values[96]-values[0]))
            for r in rows:
                if r['episode'] != 96:
                    continue
                data = {k: finite(r[k]) for k in METRICS}
                final_rows.append(dict(method=method, seed=seed, case_id=r['case_id'], **data,
                                       rule_reward=finite(rules[r['case_id']]['reward'])))
    gaps = []
    for other in METHODS[1:]:
        for seed in SEEDS:
            main0, mainf = endpoints['main', seed]
            other0, otherf = endpoints[other, seed]
            initial = main0-other0
            incremental = (mainf-main0)-(otherf-other0)
            final = mainf-otherf
            near(final, initial+incremental)
            gaps.append(dict(comparator=other, seed=seed, initial_method_gap=initial,
                             relative_learning_change=incremental, final_method_gap=final))
    grouped = {}
    for method in METHODS:
        final = [r for r in final_rows if r['method'] == method]
        means = [statistics.mean(r['reward'] for r in final if r['seed'] == s) for s in SEEDS]
        differences = [r['reward']-r['rule_reward'] for r in final]
        grouped[method] = dict(final12_mean=statistics.mean(means), final12_seed_sd=statistics.stdev(means),
            wins_against_rule=sum(d > 1e-8 for d in differences),
            ties_against_rule=sum(abs(d) <= 1e-8 for d in differences),
            losses_against_rule=sum(d < -1e-8 for d in differences),
            mean_metrics={k: statistics.mean(r[k] for r in final) for k in METRICS},
            training_frames=sum(cells[method, s]['training_frames'] for s in SEEDS),
            observed_optimizer_steps=sum(cells[method, s]['gradient_steps'] for s in SEEDS))
    decomposition = {other: {k: statistics.mean(r[k] for r in gaps if r['comparator'] == other)
        for k in ('initial_method_gap', 'relative_learning_change', 'final_method_gap')}
        for other in METHODS[1:]}
    summary = dict(contract='stream_learning_attribution_v1', seed_endpoints=seed_rows,
        paired_gap_decomposition=decomposition, final12=grouped,
        deterministic_rule_mean_metrics={k: statistics.mean(finite(r[k]) for r in rule_rows) for k in METRICS},
        interpretation='Descriptive gap identity only. Initial gaps are not randomized component effects; endpoint deletions would not establish retrained module necessity.',
        statistical_unit='Seeds nested in three consumed-training probe conditions; no independent-window significance or generalization claim.',
        scientific_interactions=0, policy_loads=0, training_updates=0, paper_goal_complete=False)
    return summary, seed_rows, gaps, final_rows


def analyze(campaign_path, audit_path, plot_path, expected_campaign_sha, expected_audit_sha, output):
    campaign_path, audit_path, plot_path, output = map(Path, (campaign_path, audit_path, plot_path, output))
    if digest(campaign_path) != expected_campaign_sha or digest(audit_path) != expected_audit_sha:
        raise ValueError('frozen campaign/audit hash differs')
    campaign, audit, plot = [json.loads(p.read_text()) for p in (campaign_path, audit_path, plot_path)]
    if (audit.get('campaign_sha256') != expected_campaign_sha or plot.get('audit_sha256') != expected_audit_sha
            or Path(plot.get('audit_path', '')).resolve() != audit_path.resolve()):
        raise ValueError('audit/plot provenance binding differs')
    summary, seeds, gaps, finals = consume(campaign, audit, plot)
    output.mkdir(parents=True, exist_ok=False)
    summary['input_sha256'] = {str(p.resolve()): digest(p) for p in (campaign_path, audit_path, plot_path)}
    for name, rows in [('seed_learning_changes.csv', seeds), ('paired_gap_decomposition.csv', gaps), ('final12_mechanisms.csv', finals)]:
        with (output/name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--plot', required=True)
    p.add_argument('--expected_campaign_sha', required=True)
    p.add_argument('--expected_audit_sha', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = analyze(args.campaign, args.audit, args.plot, args.expected_campaign_sha, args.expected_audit_sha, args.output)
    print(json.dumps({k: result[k] for k in ('contract', 'paired_gap_decomposition', 'scientific_interactions', 'paper_goal_complete')}, ensure_ascii=False))
