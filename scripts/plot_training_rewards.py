"""Plot the training runs referenced by a checkpoint manifest without training.

Keep complete seed traces and unequal run lengths. Rolling means use only full,
trailing windows; shading is seed standard deviation, not an uncertainty CI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "sa_ghmappo": "SAGH", "ppo": "PPO", "mappo": "MAPPO", "dqn": "DQN",
    "dueling_dqn": "Dueling DQN", "qmix": "QMIX", "controller_mat": "Controller MAT",
    "dag_offload_drl": "DAG Offload DRL", "cache_offload_drl": "Cache Offload DRL",
    "dt_handoff_drl": "DT Handoff DRL",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def trailing_mean(values, window):
    values = np.asarray(values, dtype=float)
    if window < 1 or len(values) < window or not np.isfinite(values).all():
        raise ValueError("Rolling mean requires finite data and a full positive window")
    result = np.full(len(values), np.nan)
    result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def load_runs(manifest_path, window):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs, sources = {}, []
    for agent, seeds in manifest.items():
        if agent.startswith("_"):
            continue
        runs[agent] = {}
        for seed, checkpoint in sorted(seeds.items(), key=lambda pair: int(pair[0])):
            checkpoint = resolve(checkpoint)
            summary_path = checkpoint.parent.parent / "train_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["agent_name"] != agent:
                raise ValueError(f"Agent mismatch: {summary_path}")
            if not summary["run_id"].endswith(f"_seed{seed}"):
                raise ValueError(f"Seed mismatch: {summary_path}")
            rows = summary.get("episode_metrics", summary.get("rows"))
            if not rows or [r["episode_index"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError(f"Missing, duplicate or unordered episodes: {summary_path}")
            if len(rows) != summary["episodes"] or len(rows) < 2 * window:
                raise ValueError(f"Incomplete/too short training history: {summary_path}")
            values = np.asarray([r["total_reward"] for r in rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite rewards: {summary_path}")
            receipts = hashlib.sha256()
            for row in rows:
                receipt_path = summary_path.parent / "episodes" / f"episode_{row['episode_index']:04d}.summary.json"
                receipt_bytes = receipt_path.read_bytes()
                receipt = json.loads(receipt_bytes)
                reward = receipt["reward_breakdown"]["total"]["sum"]
                if not math.isclose(reward, row["total_reward"], rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f"Reward differs from episode receipt: {receipt_path}")
                receipts.update(receipt_path.name.encode() + b"\0" + hashlib.sha256(receipt_bytes).digest())
            runs[agent][seed] = {"rows": rows, "values": values, "summary": summary, "root": summary_path.parent}
            sources.append({
                "agent": agent, "seed": int(seed), "run_id": summary["run_id"],
                "summary_path": str(summary_path), "summary_sha256": digest(summary_path),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "episode_receipt_chain_sha256": receipts.hexdigest(),
                "episodes": len(rows), "update_count": summary["update_count"],
                "profile": summary.get("profile"), "reward_protocol": summary.get("reward_protocol"),
                "checkpoint_selection": checkpoint.name,
            })
        lengths = {len(run["values"]) for run in runs[agent].values()}
        if len(lengths) != 1:
            raise ValueError(f"Unequal seed lengths within {agent}; refuse silent truncation")
    if not runs:
        raise ValueError("No training runs in manifest")
    return runs, sources


def summarize(values, window):
    values = np.asarray(values, dtype=float)
    mean_trace = values.mean(axis=0)
    tail = mean_trace[-2 * window:]
    return {
        "first_block_mean": float(values[:, :window].mean()),
        "previous_tail_block_mean": float(values[:, -2 * window:-window].mean()),
        "last_block_mean": float(values[:, -window:].mean()),
        "last_minus_first": float(values[:, -window:].mean() - values[:, :window].mean()),
        "last_minus_previous": float(values[:, -window:].mean() - values[:, -2 * window:-window].mean()),
        "tail_mean_trace_slope_per_episode": float(np.polyfit(np.arange(len(tail)), tail, 1)[0]),
        "tail_raw_reward_std": float(values[:, -2 * window:].std(ddof=1)),
        "convergence_proven": False,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(runs, output, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    order = [a for a in LABELS if a in runs] + sorted(set(runs) - set(LABELS))
    palette = ["#C53A38", "#236DB3", "#1C946C", "#8D68B0", "#DD8C20", "#16A1A8", "#9F6D51", "#D76B9E", "#71767D", "#737F18"]
    colors = {a: palette[i % len(palette)] for i, a in enumerate(order)}
    all_values = np.concatenate([r["values"] for seeds in runs.values() for r in seeds.values()])
    margin = max(1, float(np.ptp(all_values)) * .05)
    ylim = (all_values.min() - margin, all_values.max() + margin)

    def draw(ax, agent, seeds=False):
        matrix = np.stack([r["values"] for r in runs[agent].values()])
        x = np.arange(1, matrix.shape[1] + 1)
        smooth = np.stack([trailing_mean(v, window) for v in matrix])
        mean = smooth.mean(axis=0)
        sd = smooth.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
        if seeds:
            for v in matrix:
                ax.plot(x, v, color=colors[agent], alpha=.12, linewidth=.6)
        else:
            ax.plot(x, matrix.mean(axis=0), color=colors[agent], alpha=.2, linewidth=.8)
        ax.fill_between(x, mean - sd, mean + sd, color=colors[agent], alpha=.15, linewidth=0)
        ax.plot(x, mean, color=colors[agent], linewidth=2, label=LABELS.get(agent, agent))
        ax.set_title(f"{LABELS.get(agent, agent)} | {len(matrix)} seeds, {matrix.shape[1]} episodes", loc="left", fontsize=11)
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode total reward")
        ax.grid(alpha=.18)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=220, facecolor="white")
        fig.savefig(output / f"{name}.pdf", facecolor="white")
        plt.close(fig)

    rows = math.ceil(len(order) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.7 * rows), squeeze=False, layout="constrained")
    for ax, agent in zip(axes.flat, order):
        draw(ax, agent)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    fig.suptitle(f"Logged training reward | trailing mean {window} episodes, seed SD shading\nHistorical runs differ in training windows and budgets; descriptive only", fontsize=13)
    save(fig, "reward_curves_all_algorithms")
    for agent in order:
        fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
        draw(ax, agent, seeds=True)
        fig.suptitle(f"Raw seed traces + trailing mean ({window} episodes) +/- seed SD\nHistorical training diagnostics; convergence is not established", fontsize=10)
        save(fig, f"reward_{agent}")
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for agent in order:
        matrix = np.stack([trailing_mean(r["values"], window) for r in runs[agent].values()])
        ax.plot(np.arange(1, matrix.shape[1] + 1), matrix.mean(axis=0), color=colors[agent],
                linewidth=2.5 if agent == "sa_ghmappo" else 1.5, label=LABELS.get(agent, agent))
    ax.set(xlabel="Training episode (original lengths retained)", ylabel="Episode total reward",
           title=f"Training reward trends | {window}-episode trailing means\nDifferent training windows / budgets; not a controlled ranking")
    ax.legend(ncol=3, loc="lower right", fontsize=9)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "reward_trends_comparison")

    eval_rows, eval_sources = [], []
    for agent in order:
        for seed, run in runs[agent].items():
            path = run["root"] / "update_eval_history.json"
            if not path.exists():
                continue
            history = json.loads(path.read_text(encoding="utf-8"))
            expected_units = None
            eval_sources.append({"path": str(path), "sha256": digest(path)})
            for item in history:
                for mode, record in (("safety_projected", item), ("raw_policy", item.get("raw_policy_eval"))):
                    if record is None:
                        continue
                    if record.get("policy_evaluation_mode") != mode:
                        raise ValueError(f"Unexpected policy evaluation mode: {path}")
                    units = (tuple(record["eval_window_ids"]), tuple(record["workflow_ids"]))
                    if expected_units is None:
                        expected_units = units
                    if units != expected_units:
                        raise ValueError(f"Evaluation windows/workflows changed: {path}")
                    if not math.isfinite(record["aggregate_by_agent"][agent]["total_reward"]):
                        raise ValueError(f"Non-finite evaluation reward: {path}")
                    eval_rows.append({"agent": agent, "seed": int(seed), "update_index": item["update_index"],
                                      "episode_index": item["episode_index"], "mode": mode,
                                      "total_reward": record["aggregate_by_agent"][agent]["total_reward"]})
    if eval_rows:
        write_csv(output / "checkpoint_evaluation.csv", eval_rows)
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for agent in order:
            for mode, style in (("safety_projected", "-"), ("raw_policy", "--")):
                selected = [r for r in eval_rows if r["agent"] == agent and r["mode"] == mode]
                if not selected:
                    continue
                x = sorted({r["episode_index"] for r in selected})
                groups = [[r["total_reward"] for r in selected if r["episode_index"] == e] for e in x]
                mean = np.asarray([np.mean(g) for g in groups])
                sd = np.asarray([np.std(g, ddof=1) if len(g) > 1 else 0 for g in groups])
                ax.plot(x, mean, style, color=colors[agent], marker="o", markersize=3,
                        label=f"{LABELS.get(agent, agent)}: {mode}")
                ax.fill_between(x, mean - sd, mean + sd, color=colors[agent], alpha=.12)
        ax.set(xlabel="Training episode at checkpoint", ylabel="Fixed-window evaluation total reward",
               title="Logged checkpoint evaluations | unsmoothed mean +/- seed SD\nEarly peak and later plateau are both retained; historical window identity is unverified")
        ax.legend()
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, "checkpoint_evaluation")
    return eval_sources, eval_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rolling-window", type=int, default=16)
    args = parser.parse_args()
    if args.rolling_window < 2:
        parser.error("--rolling-window must be >= 2")
    manifest = resolve(args.manifest)
    output = resolve(args.output_root)
    if output.exists():
        parser.error("--output-root must not exist; historical results cannot be overwritten")
    import matplotlib  # Fail before publishing data if the plotting dependency is absent.
    runs, sources = load_runs(manifest, args.rolling_window)
    output.mkdir(parents=True)
    raw, aggregate, summary = [], [], []
    for agent, seeds in runs.items():
        matrix = np.stack([r["values"] for r in seeds.values()])
        smooth = np.stack([trailing_mean(v, args.rolling_window) for v in matrix])
        for seed, run in seeds.items():
            for row in run["rows"]:
                raw.append({"agent": agent, "seed": int(seed), "episode_index": row["episode_index"],
                            "total_reward": row["total_reward"], "workflow_id": row.get("workflow_id"),
                            "window_id": row.get("window_id"), "source": str(run["root"] / "train_summary.json")})
        for i in range(matrix.shape[1]):
            valid = i >= args.rolling_window - 1
            aggregate.append({"agent": agent, "episode_index": i + 1, "seed_count": len(seeds),
                              "raw_seed_mean": float(matrix[:, i].mean()),
                              "rolling_seed_mean": float(smooth[:, i].mean()) if valid else None,
                              "rolling_seed_sd": float(smooth[:, i].std(ddof=1)) if valid and len(seeds) > 1 else None})
        summary.append({"agent": agent, "seed_count": len(seeds), "episode_count_per_seed": matrix.shape[1],
                        "block_size": args.rolling_window, **summarize(matrix, args.rolling_window)})
    write_csv(output / "training_rewards_raw.csv", raw)
    write_csv(output / "training_rewards_rolling.csv", aggregate)
    write_csv(output / "tail_diagnostics.csv", summary)
    eval_sources, eval_rows = plot(runs, output, args.rolling_window)
    metadata = {
        "reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "literature_cutoff": "2026-09-28 (existing review; no new literature assessment)",
        "target_venue": "CSCWD 2027 (descriptive visualization only)",
        "artifact_run_id": manifest.parent.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "evidence_level": "RAW_TRAINING_LOGS_RECEIPT_VERIFIED_DESCRIPTIVE_ONLY",
        "confirmatory_claim": "UNVERIFIED", "training_executed": False,
        "manifest_path": str(manifest), "manifest_sha256": digest(manifest),
        "plot_script_sha256": digest(Path(__file__)),
        "command_argv": [sys.executable, *sys.argv],
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "rolling_window": args.rolling_window, "smoothing": "full trailing arithmetic mean per seed",
        "shading": "sample SD across seeds; not CI", "convergence_proven": False,
        "episode_receipts_verified": len(raw), "run_count": len(sources),
        "sources": sources, "evaluation_sources": eval_sources,
        "baseline_checkpoint_evaluation_available": sorted({r["agent"] for r in eval_rows}),
        "limitations": ["training budgets/windows/selection differ across agents",
                        "historical v70 window identity and independence are unresolved",
                        "no baseline fixed-window evaluation histories where absent; none fabricated",
                        "a training reward plateau does not prove policy convergence or algorithm superiority"],
        "outputs": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output), "runs": len(sources), "episodes_verified": len(raw), "tail_diagnostics": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
