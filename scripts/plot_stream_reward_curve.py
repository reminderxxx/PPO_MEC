"""Native-reward figures with explicit seed variation and no peak selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
STYLES = ['-', '--', '-.', ':']
LABELS = dict(main='Proposed method', sa_ghmappo='SA (legacy)',
              mappo='MAPPO (local)', ppo='PPO (local)', dqn='DQN (local)')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped(points, seeds, episodes):
    expected = {(s, e) for s in seeds for e in episodes}
    by = {}
    for row in points:
        key = (row['seed'], row['episode'])
        if key in by:
            raise ValueError('duplicate seed/episode reward point')
        value = float(row['reward'])
        if not np.isfinite(value):
            raise ValueError('nonfinite reward point')
        by[key] = value
    if set(by) != expected or len(seeds) < 2:
        raise ValueError('incomplete seed/episode grid; no interpolation or seed deletion')
    return np.array([[by[s, e] for e in episodes] for s in seeds])


def draw(ax, episodes, arrays, labels, *, raw_seeds=True):
    bounds = []
    for i, (matrix, label) in enumerate(zip(arrays, labels)):
        mean = matrix.mean(axis=0); sd = matrix.std(axis=0, ddof=1)
        if raw_seeds:
            for row in matrix:
                ax.plot(episodes, row, color=COLORS[i], alpha=.13, lw=.6)
        ax.fill_between(episodes, mean-sd, mean+sd, color=COLORS[i], alpha=.14, linewidth=0)
        ax.plot(episodes, mean, color=COLORS[i], ls=STYLES[i], lw=1.7, label=label)
        bounds.extend([matrix.min(), matrix.max(), (mean-sd).min(), (mean+sd).max()])
    ax.set_xlim(min(episodes), max(episodes)); ax.grid(alpha=.2, linewidth=.5)
    lo, hi = min(bounds), max(bounds); pad = max((hi-lo)*.08, 1.)
    ax.set_ylim(lo-pad, hi+pad)
    ax.set_xlabel('Completed training episodes')
    ax.set_ylabel('Native episodic return')
    return bounds


def export(fig, output, name, payload, caption, extra_figures=None):
    output.mkdir(parents=True, exist_ok=False)
    for title, figure in [(name, fig), *(extra_figures or [])]:
        for ext in ('png', 'pdf', 'svg'):
            figure.savefig(output/f'{title}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(figure)
    (output/'plot_data.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    (output/'caption.txt').write_text(caption+'\n')
    with (output/'seed_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['method', 'seed', 'training_episode', 'native_return'])
        for item in payload['series']:
            for seed, values in zip(payload['seeds'], item['seed_returns']):
                for episode, value in zip(payload['episodes'], values):
                    writer.writerow([item['method'], seed, episode, value])
    files = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output/'figure_manifest.json').write_text(json.dumps(dict(files=files,
        smoothing='none on fixed-policy learning curve', uncertainty='sample SD across seeds, not confidence interval',
        manuscript_promoted=False, paper_goal_complete=False), indent=2)+'\n')
    return output/f'{name}.png'


def legacy(report_path, output):
    report = json.loads(report_path.read_text()); methods = ['sa_ghmappo', 'mappo', 'ppo', 'dqn']
    seeds = [7, 17, 27]; episodes = list(range(1, 65))
    matrices = [grouped(report['legacy_training'][m]['training_points'], seeds, episodes) for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.35))
    labels = [LABELS[m] for m in methods]
    draw(axes[0], episodes, matrices, labels); axes[0].set_title('(a) Observed training returns')
    window = 8
    smooth = [np.array([[np.mean(row[max(0, i-window+1):i+1]) for i in range(len(row))]
                        for row in matrix]) for matrix in matrices]
    draw(axes[1], episodes, smooth, labels, raw_seeds=False)
    axes[1].set_title('(b) Same data, trailing 8-episode mean')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, .99), fontsize=8)
    fig.text(.5, .005, 'Legacy sampled training return (3 seeds); bands: mean +/- 1 sample SD. No generalization or convergence claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, .90])
    payload = dict(contract='legacy_training_return_diagnostic_v1', report=str(report_path),
        report_sha256=digest(report_path), seeds=seeds, episodes=episodes,
        series=[dict(method=m, seed_returns=matrix.tolist()) for m, matrix in zip(methods, matrices)],
        smoothing=dict(left='none', right='trailing arithmetic mean, window8, min_periods1'),
        current_intent_method=False, source_version='35975cb legacy policy; not the later intent candidate')
    caption = ('Legacy sampled training returns of the original SA, MAPPO, PPO and DQN local stream controllers. '
        'Each method uses all three recorded seeds and the same per-episode exogenous case schedule. '
        'Panel (a) retains unsmoothed episode returns and faint individual seed traces. Panel (b) applies the same '
        'predeclared trailing-eight arithmetic mean to every seed/method, including shorter prefixes. '
        'Lines are seed means and shading is +/- one sample SD after the stated transformation; neither is a confidence interval. '
        'Changing training cases affect episode returns. These records are the legacy policy version, not the current intent method, '
        'and do not establish fixed-policy convergence, independent generalization or paper readiness.')
    return export(fig, output, 'legacy_training_return', payload, caption)


def fixed_policy(data_path, output):
    payload = json.loads(data_path.read_text())
    if payload.get('contract') != 'stream_reward_curve_plot_v1' or payload.get('selection') != 'fixed_final_no_peak_selection':
        raise ValueError('unqualified plot contract/selection')
    audit = Path(payload['audit_path'])
    evidence = json.loads(audit.read_text())
    if digest(audit) != payload['audit_sha256'] or evidence['status'] != 'curve_receipts_and_checkpoints_audited':
        raise ValueError('plot requires completed bound audit')
    methods = ['main', 'mappo', 'ppo', 'dqn']; seeds = [7, 17, 27, 37, 47]
    episodes = list(range(0, 97, 8))
    if (payload['seeds'] != seeds or payload['episodes'] != episodes
            or [s['method'] for s in payload['series']] != methods):
        raise ValueError('figure cannot omit/reorder a method, seed or checkpoint')
    matrices = []
    by = {(c['method'], c['seed']):c for c in evidence['cells']}
    if len(evidence['cells']) != 20 or set(by) != {(m,s) for m in methods for s in seeds}:
        raise ValueError('audited seed/method pool differs')
    probes = evidence['probe_case_ids']
    if len(probes) != 3 or len(set(probes)) != 3: raise ValueError('fixed probe pool differs')
    for item in payload['series']:
        matrix = np.asarray(item['seed_returns'], dtype=float)
        if matrix.shape != (len(seeds), len(episodes)) or not np.isfinite(matrix).all():
            raise ValueError('incomplete fixed-policy curve; no interpolation')
        for i, seed in enumerate(seeds):
            cell = by[item['method'], seed]
            cp = {r['episode']:r for r in cell['checkpoints']}
            for j, ep in enumerate(episodes):
                selected = [r for r in cell['evaluation'] if r['episode']==ep and r['case_id'] in probes]
                if len(selected) != 3 or {r['case_id'] for r in selected} != set(probes):
                    raise ValueError('audited fixed probe points missing')
                if abs(matrix[i,j]-np.mean([r['reward'] for r in selected])) > 1e-8:
                    raise ValueError('plot reward differs from audited native evaluation')
                if item['seed_frames'][i][j] != cp[ep]['cumulative_training_frames']:
                    raise ValueError('plot physical training frame axis differs')
        matrices.append(matrix)
    fig, ax = plt.subplots(figsize=(7.16, 3.65))
    bounds = draw(ax, episodes, matrices, [LABELS[m] for m in methods])
    value = float(payload['gain_rule_probe_mean'])
    if not np.isfinite(value): raise ValueError('nonfinite rule reference')
    rule_rows = [r for r in evidence['rules'] if r['case_id'] in probes]
    if len(rule_rows) != 3 or abs(value-np.mean([r['reward'] for r in rule_rows])) > 1e-8:
        raise ValueError('rule reference differs from audited native evaluation')
    ax.axhline(value, color='#333333', lw=1.2, ls=(0, (6, 2)), label='Frozen gain rule')
    lo, hi = min([*bounds, value]), max([*bounds, value]); pad = max(1., (hi-lo)*.08)
    ax.set_ylim(lo-pad, hi+pad)
    ax.set_ylabel('Deterministic policy return')
    ax.legend(frameon=False, ncol=2, fontsize=8, loc='best')
    fig.text(.5, .005, '5 seeds; fixed consumed-train probes; mean +/- 1 sample SD; no smoothing.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    # Actual frames differ after real early termination. Draw every seed at its
    # measured budget; do not interpolate seeds onto a fabricated common grid.
    frame_fig, frame_ax = plt.subplots(figsize=(7.16, 3.65))
    for i, (item, matrix) in enumerate(zip(payload['series'], matrices)):
        for j, values in enumerate(matrix):
            frames = item['seed_frames'][j]
            if len(frames) != len(episodes) or frames != sorted(set(frames)):
                raise ValueError('actual frame axis missing or nonincreasing')
            frame_ax.plot(frames, values, color=COLORS[i], ls=STYLES[i], lw=1., alpha=.5,
                          label=LABELS[item['method']] if j == 0 else None)
    frame_ax.axhline(value, color='#333333', lw=1.2, ls=(0,(6,2)), label='Frozen gain rule')
    frame_ax.set_xlabel('Actual training environment frames')
    frame_ax.set_ylabel('Deterministic policy return'); frame_ax.grid(alpha=.2, linewidth=.5)
    frame_ax.legend(frameon=False,ncol=2,fontsize=8)
    frame_fig.text(.5,.005,'Every line is one seed at its actual frame budget; no aligned-budget averaging.',ha='center',fontsize=8)
    frame_fig.tight_layout(rect=[0,.04,1,1])
    caption = ('Learning curves from randomly initialized policies with a shared frozen, previously fitted GRU predictor. '
        'At episode0 and every8 completed training episodes, each policy is evaluated greedily on the same three predeclared '
        'consumed-training conditions. Each plotted seed value averages those conditions; lines and shading show the mean and '
        '+/- one sample SD across all five seeds. Faint lines show every seed. No smoothing, reward rescaling, seed removal, '
        'checkpoint selection or extrapolation is applied. The frozen gain-rule reference uses the same three conditions once '
        'and has no invented seed variance. At the fixed final episode96, all12 declared conditions are evaluated; their summary '
        'is reported separately and is not substituted into the three-probe curve. The x-axis is completed exogenous sessions; '
        'actual physical frames, gradients, parameter counts, predictor exposure and compute costs are separately retained. '
        'These consumed-training learning curves do not certify convergence, independent generalization, external-paper fidelity '
        'or publication acceptance. All negative seeds and checkpoint regressions remain visible.')
    return export(fig, output, 'reward_learning_curve', payload, caption,
                  extra_figures=[('reward_vs_actual_frames', frame_fig)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy_report', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.legacy_report) == bool(args.data): raise ValueError('choose exactly one source')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})
    print(legacy(args.legacy_report, args.output) if args.legacy_report else fixed_policy(args.data, args.output))


if __name__ == '__main__': main()
