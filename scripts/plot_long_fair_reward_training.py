"""Create complete, auditable episode learning curves for multi-cycle runs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from plot_fair_reward_training import LABELS, read_csv, sha, tail_diagnostic, write_csv


def trailing_mean(values, window=100):
    values = np.asarray(values, dtype=float)
    cumulative = np.r_[0., np.cumsum(values)]
    end = np.arange(1, len(values)+1)
    start = np.maximum(0, end-window)
    return (cumulative[end]-cumulative[start])/(end-start)


def validate_cell(cell, p, name, seed):
    c = json.loads((cell / "complete.json").read_text())
    if c["status"] != "COMPLETED" or c["plan_sha256"] != p["plan_sha256"] or c["training_steps"] != p["training_steps"]:
        raise ValueError("Incomplete or mismatched cell")
    if c["updates"] != p["training_steps"]//p["steps_per_update"] or c["cycles"] != p["training_cycles"]:
        raise ValueError("Incorrect cycle/update budget")
    for filename, key in [("training.csv", "training_csv_sha256"), ("evaluation.csv", "evaluation_csv_sha256"),
                          ("episode_receipts.jsonl", "episode_receipts_sha256"), ("updates.jsonl", "updates_sha256")]:
        if sha(cell / filename) != c[key]:
            raise ValueError(f"Payload changed: {cell / filename}")
    rows = read_csv(cell / "training.csv")
    exposure, total = defaultdict(int), 0
    train_units = [(w["window_id"], wf) for wf in p["workflow_ids"] for w in p["windows"] if w["split"] == "train"]
    if len(rows) != c["episodes"]:
        raise ValueError("Episode count mismatch")
    with (cell / "episode_receipts.jsonl").open() as receipts:
        for i, row in enumerate(rows, 1):
            receipt = json.loads(next(receipts))
            n = int(row["episode_steps"])
            total += n
            unit = int(row["unit_index"])
            cycle = unit//len(train_units)+1
            if int(row["episode_index"]) != i or int(row["training_steps"]) != total or int(row["cycle"]) != cycle:
                raise ValueError("Episode sequence or cycle mismatch")
            if (row["window_id"], row["workflow_id"]) != train_units[unit % len(train_units)]:
                raise ValueError("Training unit schedule mismatch")
            meta = receipt["metadata"]
            if meta["agent"] != name or meta["seed"] != seed or meta["plan_sha256"] != p["plan_sha256"] or meta["unit_index"] != unit:
                raise ValueError("Receipt identity mismatch")
            reward = receipt["reward_breakdown"]
            reconstructed = sum(reward[k]["sum"] for k in ("service_reward", "continuity_bonus", "mechanism_exploration_bonus", "positive_offset"))
            reconstructed -= sum(reward[k]["sum"] for k in ("delay_penalty", "cache_miss_penalty", "migration_cost", "constraint_penalty"))
            if not np.isclose(reconstructed, float(row["total_reward"]), atol=1e-5, rtol=0) or reward["positive_offset"]["sum"] != 0:
                raise ValueError("Native reward accounting mismatch")
            if not np.isclose(reward["total"]["sum"], float(row["total_reward"]), atol=1e-6, rtol=0):
                raise ValueError("Episode return mismatch")
            if n != receipt["episode_status"]["total_steps"] or n != len(receipt["actions"]) or receipt["training_steps"] != total:
                raise ValueError("Receipt transition count mismatch")
            exposure[unit] += n
        if next(receipts, None) is not None:
            raise ValueError("Extra receipts")
    if total != p["training_steps"] or set(exposure) != set(range(len(train_units)*p["training_cycles"])) or set(exposure.values()) != {p["steps_per_training_unit"]}:
        raise ValueError("Unequal training exposure")
    with (cell / "updates.jsonl").open() as stream:
        updates = [json.loads(line) for line in stream]
    if [r["training_steps"] for r in updates] != list(range(p["steps_per_update"], total+1, p["steps_per_update"])):
        raise ValueError("Incorrect learner update opportunities")
    if [r["update"] for r in updates] != list(range(1, c["updates"]+1)):
        raise ValueError("Incorrect update sequence")
    evaluation = read_csv(cell / "evaluation.csv")
    points = list(range(0, total+1, p["steps_per_update"]*p["evaluate_every_updates"]))
    units = {(w["window_id"], wf) for w in p["windows"] if w["split"] == "validation" for wf in p["workflow_ids"]}
    combined, means, completion = [], [], []
    for step in points:
        sub = read_csv(cell / f"evaluation_{step:07d}.csv")
        if len(sub) != len(units) or {(r["window_id"], r["workflow_id"]) for r in sub} != units:
            raise ValueError("Validation units differ")
        if any(r["agent"] != name or int(r["seed"]) != seed or int(r["training_steps"]) != step for r in sub):
            raise ValueError("Validation identity differs")
        combined.extend(sub)
        means.append(float(np.mean([float(r["total_reward"]) for r in sub])))
        completion.append(float(np.mean([r["completed"] == "True" for r in sub])))
        with (cell / f"validation_actions_{step:07d}.jsonl").open() as stream:
            trace = [json.loads(line) for line in stream]
        actual = defaultdict(float)
        for row in trace:
            actual[row["window_id"], row["workflow_id"]] += float(row["reward"])
        if set(actual) != units or any(not np.isclose(actual[r["window_id"], r["workflow_id"]], float(r["total_reward"]), atol=1e-6, rtol=0) for r in sub):
            raise ValueError("Executed validation rewards do not match CSV")
    if combined != evaluation:
        raise ValueError("Aggregate validation CSV differs")
    return rows, evaluation, means, completion


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    out = root / "figures"
    if out.exists():
        raise FileExistsError("Figure export is create-only")
    p = json.loads((root / "frozen_plan.json").read_text())
    all_training, all_evaluation, diagnostics, raw = [], [], [], {}
    validation, training, workflow_curves = {}, {}, {}
    curves_csv, action_diagnostics = [], []
    for name in p["agents"]:
        validation[name], training[name] = [], []
        workflow_curves[name] = {wf: [] for wf in p["workflow_ids"]}
        raw[name] = []
        for seed in p["seeds"]:
            cell = root / "cells" / name / f"seed{seed}"
            rows, evaluations, means, completion = validate_cell(cell, p, name, seed)
            rewards = [float(r["total_reward"]) for r in rows]
            rolling = trailing_mean(rewards, p["plotting"]["episode_trailing_mean"])
            training[name].append(rolling)
            raw[name].append(rewards)
            validation[name].append(means)
            for wf in p["workflow_ids"]:
                workflow_curves[name][wf].append(trailing_mean([float(r["total_reward"]) for r in rows if r["workflow_id"] == wf], 100))
            for row, value in zip(rows, rolling):
                curves_csv.append({"agent": name, "seed": seed, "episode_index": row["episode_index"],
                                   "training_steps": row["training_steps"], "raw_total_reward": row["total_reward"],
                                   "trailing_mean_100_episodes": float(value), "budget_cut": row["budget_cut"]})
            diagnostics.append({"agent": name, "seed": seed, **tail_diagnostic(means, completion, p["convergence"])})
            all_training.extend(rows)
            all_evaluation.extend(evaluations)
            if name == "sa_ghmappo":
                paired = read_csv(cell / "paired_initial_states.csv")
                for step in sorted({int(r["training_steps"]) for r in paired}):
                    sub = [r for r in paired if int(r["training_steps"]) == step]
                    action_diagnostics.append({"seed": seed, "training_steps": step, "states": len(sub),
                                               "actual_action_changes": sum(r["actual_action_changed"] == "True" for r in sub),
                                               "network_argmax_changes": sum(r["network_argmax_changed"] == "True" for r in sub),
                                               "network_probability_l1_mean": float(np.mean([float(r["network_probability_l1"]) for r in sub]))})
    out.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    colors = plt.get_cmap("tab10").colors

    def save(fig, name):
        fig.savefig(out / (name+".png"), dpi=220, facecolor="white")
        fig.savefig(out / (name+".pdf"), facecolor="white")
        plt.close(fig)

    def draw(ax, curves, color, label, x=None):
        # Every seed's entire observed trajectory stays visible. The ensemble
        # mean/SD stop at the common observed prefix, without tail padding.
        for curve in curves:
            ax.plot(np.arange(1, len(curve)+1) if x is None else x, curve, color=color, alpha=.22, linewidth=.55)
        length = min(map(len, curves))
        matrix = np.asarray([r[:length] for r in curves])
        axis = np.arange(1, length+1) if x is None else x[:length]
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
        ax.plot(axis, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(axis, mean-sd, mean+sd, color=color, alpha=.09, linewidth=0)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)

    points = list(range(0, p["training_steps"]+1, p["steps_per_update"]*p["evaluate_every_updates"]))
    for kind, curves in [("episode_reward", training), ("fixed_validation_reward", validation)]:
        fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
        for i, name in enumerate(p["agents"]):
            draw(ax, curves[name], colors[i], LABELS[name], points if kind.startswith("fixed") else None)
        ax.set(xlabel="Training environment interactions" if kind.startswith("fixed") else "Training episode",
               ylabel="Validation episode total reward" if kind.startswith("fixed") else "Episode total reward (trailing mean, 100 episodes)",
               title="Fixed validation; all checkpoints, no smoothing" if kind.startswith("fixed") else f"Training learning curves; common interaction budget and {p['training_cycles']} complete cycles")
        ax.legend(ncol=3, fontsize=8)
        save(fig, kind+"_all_algorithms")
    fig, axes = plt.subplots(len(p["workflow_ids"]), 1, figsize=(11, 5*len(p["workflow_ids"])), squeeze=False, layout="constrained")
    for ax, wf in zip(axes[:, 0], p["workflow_ids"]):
        for i, name in enumerate(p["agents"]):
            draw(ax, workflow_curves[name][wf], colors[i], LABELS[name])
        ax.set(title=f"Workflow {wf}: training episode return", xlabel="Training episode within workflow",
               ylabel="Episode total reward (trailing mean, 100 episodes)")
        ax.legend(ncol=3, fontsize=8)
    save(fig, "episode_reward_by_workflow")
    for i, name in enumerate(p["agents"]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for rewards in raw[name]:
            axes[0].plot(np.arange(1, len(rewards)+1), rewards, color=colors[i], alpha=.05, linewidth=.35)
        draw(axes[0], training[name], colors[i], LABELS[name])
        draw(axes[1], validation[name], colors[i], LABELS[name], points)
        axes[0].set(title="All raw returns + 100-episode trailing mean", xlabel="Training episode", ylabel="Episode total reward")
        axes[1].set(title="Fixed validation, unsmoothed", xlabel="Training environment interactions", ylabel="Validation episode total reward")
        fig.suptitle(f"{LABELS[name]}: all seeds; mean +/- seed SD; unequal episode counts are not padded")
        save(fig, "learning_curve_"+name)
    write_csv(out / "all_training_episodes.csv", all_training)
    write_csv(out / "all_validation_units.csv", all_evaluation)
    write_csv(out / "episode_curve_values.csv", curves_csv)
    write_csv(out / "convergence_diagnostics.csv", diagnostics)
    if action_diagnostics:
        write_csv(out / "sagh_paired_action_summary.csv", action_diagnostics)
    report = {"artifact_run_id": root.name, "plan_sha256": p["plan_sha256"], "source_commit": p["source_commit"],
              "evidence_level": "MATCHED_DEVELOPMENT_NATIVE_EPISODE_RECEIPTS_AND_ACTION_TRACES", "formal_holdout": "UNVERIFIED",
              "plot_rules": p["plotting"], "all_seeds_retained": True,
              "training_axis_note": "Same interactions/updates/cycles; actual episode counts differ. Mean/SD only on common observed prefix; all full seed curves remain visible.",
              "diagnostics": diagnostics, "files": {f.name: sha(f) for f in out.iterdir()}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "EXPORTED", "figures": str(out), "cells": len(diagnostics)}))


if __name__ == "__main__":
    main()
