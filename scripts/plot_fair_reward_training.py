"""Export all fixed-budget results without selection, interpolation or padding."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = {"sa_ghmappo": "SAGH", "ppo": "PPO", "mappo": "MAPPO", "dqn": "DQN",
          "dueling_dqn": "Dueling DQN", "qmix": "QMIX", "controller_mat": "Controller MAT",
          "dag_offload_drl": "DAG Offload DRL", "cache_offload_drl": "Cache Offload DRL", "dt_handoff_drl": "DT Handoff DRL"}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def tail_diagnostic(rewards, completions, rules):
    tail = np.asarray(rewards[-rules["tail_evaluation_points"]:], dtype=float)
    scale = max(1., abs(float(tail.mean())))
    relative_range = float(np.ptp(tail) / scale)
    relative_slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0] / scale)
    return {"tail_mean": float(tail.mean()), "tail_relative_range": relative_range,
            "tail_relative_slope_per_checkpoint": relative_slope,
            "tail_points": len(tail),
            "reward_plateau": len(tail) >= rules["tail_evaluation_points"] and relative_range <= rules["relative_range_max"] and abs(relative_slope) <= rules["relative_slope_max"],
            "tail_completion_rate": float(np.mean(completions[-len(tail):])),
            "completion_floor_met": float(np.mean(completions[-len(tail):])) >= rules["minimum_tail_completion_rate"],
            "first_evaluation_reward": float(rewards[0]), "final_evaluation_reward": float(rewards[-1]),
            "final_minus_initial_reward": float(rewards[-1] - rewards[0]), "policy_convergence_proven": False}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(); root = args.run_root.resolve()
    p = json.loads((root / "frozen_plan.json").read_text())
    out = root / "figures"
    if out.exists():
        raise FileExistsError("Refusing to replace an earlier figure export")
    expected_points = list(range(0, p["training_steps"] + 1, p["steps_per_update"] * p["evaluate_every_updates"]))
    curves, training, raw_training, raw_evaluation, diagnostics, inventory = {}, {}, [], [], [], []
    for name in p["agents"]:
        curves[name], training[name] = [], []
        for seed in p["seeds"]:
            cell = root / "cells" / name / f"seed{seed}"
            complete = json.loads((cell / "complete.json").read_text())
            if complete["training_steps"] != p["training_steps"] or complete["plan_sha256"] != p["plan_sha256"]:
                raise ValueError("Cell budget/plan mismatch")
            for basename in ("training.csv", "evaluation.csv"):
                if sha(cell / basename) != complete[basename.replace(".csv", "_csv_sha256")]:
                    raise ValueError(f"Cell payload changed: {cell / basename}")
            rows, evaluations = read_csv(cell / "training.csv"), read_csv(cell / "evaluation.csv")
            exposure = defaultdict(int)
            for r in rows:
                exposure[r["window_id"], r["workflow_id"]] += int(r["episode_steps"])
            if len(exposure) != p["train_window_count"] * p["max_workflows"] or set(exposure.values()) != {p["steps_per_training_unit"]}:
                raise ValueError("Window/workflow exposure differs from common budget")
            if sum(int(r["episode_steps"]) for r in rows) != p["training_steps"]:
                raise ValueError("Wrong native interaction count")
            means, completions = [], []
            if sorted({int(r["training_steps"]) for r in evaluations}) != expected_points:
                raise ValueError("Missing evaluation checkpoint")
            for step in expected_points:
                selected = [r for r in evaluations if int(r["training_steps"]) == step]
                expected_units = {(w["window_id"], wf) for w in p["windows"] if w["split"] == "validation" for wf in p["workflow_ids"]}
                if len(selected) != len(expected_units) or {(r["window_id"], r["workflow_id"]) for r in selected} != expected_units:
                    raise ValueError("Evaluation unit mismatch")
                means.append(float(np.mean([float(r["total_reward"]) for r in selected])))
                completions.append(float(np.mean([r["completed"] == "True" for r in selected])))
            curves[name].append(means)
            # Sum native episode returns / actual steps in a fixed interaction block.
            # Budget-cut episodes remain explicit in the raw CSV, never extended.
            block_reward, block_steps = defaultdict(float), defaultdict(int)
            for r in rows:
                index = (int(r["training_steps"]) - 1) // p["steps_per_update"]
                block_reward[index] += float(r["total_reward"])
                block_steps[index] += int(r["episode_steps"])
            if any(count != p["steps_per_update"] for count in block_steps.values()):
                raise ValueError("Incomplete update reward block")
            training[name].append([block_reward[i] / block_steps[i] for i in sorted(block_reward)])
            diagnostics.append({"agent": name, "seed": seed, **tail_diagnostic(means, completions, p["convergence"])})
            raw_training.extend(rows); raw_evaluation.extend(evaluations)
            inventory.append({"cell": str(cell), "complete_sha256": sha(cell / "complete.json"),
                              "initial_checkpoint_sha256": sha(cell / "initial.pt"), "final_checkpoint_sha256": sha(cell / f"update_{complete['updates']:04d}.pt")})
    out.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    colors = plt.get_cmap("tab10").colors
    curve_rows = []
    for i, name in enumerate(p["agents"]):
        for j, step in enumerate(expected_points):
            values = np.asarray(curves[name])[:, j]
            curve_rows.append({"agent": name, "training_steps": step, "seed_count": len(values),
                               "evaluation_total_reward_mean": float(values.mean()), "evaluation_total_reward_seed_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.})

    def save(fig, basename):
        fig.savefig(out / f"{basename}.png", dpi=220, facecolor="white")
        fig.savefig(out / f"{basename}.pdf", facecolor="white")
        plt.close(fig)

    def draw(ax, matrix, x, color, label, raw=False):
        matrix = np.asarray(matrix)
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
        if raw:
            for values in matrix:
                ax.plot(x, values, color=color, alpha=.22, linewidth=.7)
        ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=2.5, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.13, linewidth=0)
        ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for i, name in enumerate(p["agents"]):
        draw(ax, curves[name], expected_points, colors[i], LABELS[name])
    ax.set(title="Fair training: fixed-window evaluation reward\nFresh initialization, common windows/reward/interactions; unsmoothed mean +/- seed SD",
           xlabel="Training environment interactions", ylabel="Evaluation episode total reward")
    ax.legend(ncol=3, fontsize=9)
    save(fig, "fair_evaluation_reward_all_algorithms")
    for i, name in enumerate(p["agents"]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        draw(axes[0], training[name], np.arange(1, len(training[name][0]) + 1) * p["steps_per_update"], colors[i], LABELS[name], raw=True)
        draw(axes[1], curves[name], expected_points, colors[i], LABELS[name], raw=True)
        axes[0].set(title=f"{LABELS[name]}: native training reward", xlabel="Environment interactions", ylabel=f"Native reward per step ({p['steps_per_update']}-step block)")
        axes[1].set(title="Fixed validation windows; no smoothing", xlabel="Training environment interactions", ylabel="Evaluation episode total reward")
        fig.suptitle(f"{len(p['seeds'])} seeds; faint seed traces, mean and seed SD; no curve alignment or padding", fontsize=10)
        save(fig, f"fair_reward_{name}")
    write_csv(out / "all_training_episodes.csv", raw_training)
    write_csv(out / "all_evaluation_units.csv", raw_evaluation)
    write_csv(out / "evaluation_reward_summary.csv", curve_rows)
    write_csv(out / "convergence_diagnostics.csv", diagnostics)
    report = {"reviewed_at": "2026-10-21", "literature_cutoff": "2026-09-28 (not reassessed)",
              "target_venue": "CSCWD 2027, development experiment only", "artifact_run_id": root.name,
              "policy_version": "tmc_review_policy_v3_20260621", "git_commit": p["source_commit"],
              "plan_sha256": p["plan_sha256"], "evidence_level": "MATCHED_DEVELOPMENT_TRAINING_AND_VALIDATION",
              "formal_or_holdout_claim": "UNVERIFIED", "cells": inventory,
              "diagnostics": diagnostics, "curve_adjustment": "none; all original validation points retained",
              "limitations": ["development validation uses previously explored data, not untouched holdout",
                              "native algorithm architectures, hyperparameters and per-update gradient work differ",
                              "reward plateau is descriptive and does not prove policy convergence"],
              "outputs": {path.name: sha(path) for path in sorted(out.iterdir())}}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"figures": str(out), "cells_verified": len(inventory), "reward_plateau_seeds": sum(r["reward_plateau"] for r in diagnostics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
