"""Decompose saved reward gaps, retaining every baseline, seed and checkpoint.

Nonzero-return and action-disagreement subsets are descriptive diagnostics.
They are never used to replace full-population results or prove superiority.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from zoneinfo import ZoneInfo


TOLERANCE = 1e-8
LABELS = {"sa_ghmappo": "SA-GHMAPPO", "ppo": "PPO", "mappo": "MAPPO", "dqn": "DQN",
          "dueling_dqn": "Dueling DQN", "qmix": "QMIX", "controller_mat": "Controller MAT",
          "dag_offload_drl": "DAG offload DRL", "cache_offload_drl": "Cache offload DRL",
          "dt_handoff_drl": "DT handoff DRL"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decompose(rows):
    require(bool(rows), "Missing paired population")
    deltas = [float(row["delta"]) for row in rows]
    require(all(math.isfinite(v) for v in deltas), "Nonfinite reward difference")
    selected = [v for v in deltas if abs(v) > TOLERANCE]
    rest = [v for v in deltas if abs(v) <= TOLERANCE]
    different_actions = [row["delta"] for row in rows if row["different_actions"]]
    count = len(rows)
    total = math.fsum(deltas) / count
    weighted = math.fsum(selected) / count
    remainder = math.fsum(rest) / count
    require(abs(total - weighted - remainder) < 1e-10, "Gap decomposition failed")
    return {"paired_units": count, "different_reward_units": len(selected),
            "different_reward_fraction": len(selected) / count,
            "full_gap": total,
            "nonzero_reward_conditional_gap": statistics.mean(selected) if selected else None,
            "nonzero_reward_weighted_contribution": weighted,
            "remaining_weighted_contribution": remainder,
            "different_action_units": len(different_actions),
            "different_action_fraction": len(different_actions) / count,
            "different_action_conditional_gap": statistics.mean(different_actions) if different_actions else None,
            "different_actions_equal_reward_units": sum(row["different_actions"] and abs(row["delta"]) <= TOLERANCE for row in rows),
            "same_actions_different_reward_units": sum(not row["different_actions"] and abs(row["delta"]) > TOLERANCE for row in rows),
            "positive_units": sum(v > TOLERANCE for v in deltas),
            "negative_units": sum(v < -TOLERANCE for v in deltas)}


def trend_summary(values, main_rewards):
    require(len(values) == len(main_rewards) and len(values) >= 5, "Insufficient longitudinal points")
    tail = values[-5:]
    return {"initial_full_gap": values[0], "final_full_gap": values[-1],
            "final_minus_initial_gap": values[-1] - values[0],
            "gap_never_decreased_at_recorded_points": all(b >= a - TOLERANCE for a, b in zip(values, values[1:])),
            "final_gap_positive": values[-1] > TOLERANCE,
            "tail_five_gap_range": max(tail) - min(tail),
            "main_reward_final_minus_initial": main_rewards[-1] - main_rewards[0],
            "main_reward_improved": main_rewards[-1] > main_rewards[0] + TOLERANCE,
            "policy_convergence_proven": False,
            "formal_superiority": "UNVERIFIED"}


def reward_groups(names, signature):
    groups = []
    for name in names:
        for group in groups:
            if signature[name] == signature[group[0]]:
                group.append(name)
                break
        else:
            groups.append([name])
    return groups


def export(root, audit_path, out):
    require(not out.exists(), "Output is create-only")
    plan = json.loads((root / "frozen_plan.json").read_text())
    expected = plan["plan_sha256"]
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    require(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() == expected,
            "Frozen plan changed")
    audit = json.loads(audit_path.read_text())
    require(audit["checks"]["status"] == "PASS" and audit["artifact_run_id"] == root.name
            and audit["git_commit"] == plan["source_commit"], "Mismatched input audit")
    points = list(range(0, plan["training_steps"] + 1, plan["steps_per_update"] * plan["evaluate_every_updates"]))
    units = sorted((w["window_id"], wf) for w in plan["windows"] if w["split"] == "validation" for wf in plan["workflow_ids"])
    records, source_hashes = {}, {str(audit_path): sha(audit_path), str(root / "frozen_plan.json"): sha(root / "frozen_plan.json")}
    steps = files = 0
    for name in plan["agents"]:
        for seed in plan["seeds"]:
            cell = root / "cells" / name / f"seed{seed}"
            path = cell / "evaluation.csv"
            actual_hash = sha(path)
            require(actual_hash == audit["source_files_sha256"][str(path)], "Evaluation changed since audit")
            source_hashes[str(path)] = actual_hash
            with path.open(newline="") as stream:
                evaluations = list(csv.DictReader(stream))
            indexed = {}
            for row in evaluations:
                key = int(row["training_steps"]), (row["window_id"], row["workflow_id"])
                require(row["agent"] == name and int(row["seed"]) == seed and key not in indexed, "Duplicate or mismatched evaluation")
                require(math.isfinite(float(row["total_reward"])), "Nonfinite recorded return")
                indexed[key] = row
            require(set(indexed) == {(t, u) for t in points for u in units}, "Missing evaluation point/unit")
            for point in points:
                path = cell / f"validation_actions_{point:07d}.jsonl"
                actual_hash = sha(path)
                require(actual_hash == audit["source_files_sha256"][str(path)], "Action trace changed since audit")
                source_hashes[str(path)] = actual_hash
                per_unit = defaultdict(dict)
                with path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        unit = row["window_id"], row["workflow_id"]
                        step = int(row["step"])
                        require(step not in per_unit[unit], "Duplicate trace step")
                        require(math.isfinite(float(row["reward"])), "Nonfinite step reward")
                        per_unit[unit][step] = int(row["action"]), float(row["reward"])
                        steps += 1
                files += 1
                require(set(per_unit) == set(units), "Trace unit population differs")
                for unit, trace in per_unit.items():
                    require(set(trace) == set(range(len(trace))), "Missing trace step")
                    actions = tuple(trace[i][0] for i in range(len(trace)))
                    reward = float(indexed[point, unit]["total_reward"])
                    require(abs(math.fsum(v[1] for v in trace.values()) - reward) <= 1e-6, "Trace return differs")
                    records[name, seed, point, unit] = {"actions": actions, "reward": reward}

    curves = {name: [statistics.mean(records[name, s, t, u]["reward"] for s in plan["seeds"] for u in units)
                     for t in points] for name in plan["agents"]}
    signatures = {name: tuple(records[name, s, t, u]["reward"] for s in plan["seeds"] for t in points for u in units)
                  for name in plan["agents"]}
    groups = reward_groups(plan["agents"], signatures)
    pairs, summaries, trends = [], [], []
    for baseline in [name for name in plan["agents"] if name != "sa_ghmappo"]:
        initial_cohort = {(s, u) for s in plan["seeds"] for u in units
                          if records["sa_ghmappo", s, 0, u]["actions"] != records[baseline, s, 0, u]["actions"]}
        for point in points:
            pooled = []
            for seed in plan["seeds"]:
                seed_rows = []
                for unit in units:
                    main, other = records["sa_ghmappo", seed, point, unit], records[baseline, seed, point, unit]
                    row = {"baseline": baseline, "seed": seed, "training_steps": point,
                           "window_id": unit[0], "workflow_id": unit[1],
                           "different_actions": main["actions"] != other["actions"],
                           "delta": main["reward"] - other["reward"],
                           "in_initial_behavior_cohort": (seed, unit) in initial_cohort}
                    pairs.append(row)
                    seed_rows.append(row)
                summaries.append({"baseline": baseline, "seed": str(seed), "training_steps": point, **decompose(seed_rows)})
                pooled.extend(seed_rows)
            fixed = [row["delta"] for row in pooled if row["in_initial_behavior_cohort"]]
            summaries.append({"baseline": baseline, "seed": "pooled", "training_steps": point,
                              **decompose(pooled), "initial_behavior_cohort_units": len(fixed),
                              "initial_behavior_cohort_gap": statistics.mean(fixed) if fixed else None})
        values = [row["full_gap"] for row in summaries if row["baseline"] == baseline and row["seed"] == "pooled"]
        trends.append({"baseline": baseline, **trend_summary(values, curves["sa_ghmappo"])})

    # Longitudinal action groups are recorded without treating algorithms as
    # equivalent outside these saved episodes or dropping any group member.
    behavior_groups = []
    for seed in plan["seeds"]:
        for point in points:
            action_signatures = {name: tuple(records[name, seed, point, u]["actions"] for u in units) for name in plan["agents"]}
            clustered = reward_groups(plan["agents"], action_signatures)
            behavior_groups.append({"seed": seed, "training_steps": point, "unique_trajectory_groups": len(clustered),
                                    "members": json.dumps(clustered)})
    out.mkdir(parents=True)
    for filename, rows in [("all_paired_units.csv", pairs), ("decomposition_by_checkpoint.csv", summaries),
                           ("gap_trends.csv", trends), ("behavior_groups_by_checkpoint.csv", behavior_groups)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    render(out, plan, points, curves, groups, summaries)
    report = {"reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "literature_cutoff": "2026-09-28 (not reassessed)", "target_venue": "CSCWD2027 development",
              "artifact_run_id": root.name, "policy_version": "tmc_review_policy_v3_20260621",
              "git_commit": plan["source_commit"], "workspace_git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "evidence_level": "HASH_BOUND_ALL_CHECKPOINT_PAIRED_DIFFERENCE_DECOMPOSITION_ONLY",
              "formal_superiority": "UNVERIFIED", "policy_convergence_proven": False,
              "checks": {"trace_files": files, "trace_steps": steps, "paired_records": len(pairs),
                         "all_baselines": len(trends), "all_seeds": plan["seeds"], "all_checkpoints": len(points),
                         "new_environment_steps": 0, "new_optimizer_steps": 0},
              "identical_full_unit_reward_groups": groups, "trends": trends,
              "final_decomposition": [row for row in summaries if row["seed"] == "pooled" and row["training_steps"] == points[-1]],
              "definitions": {"difference_tolerance": TOLERANCE,
                              "nonzero_reward_subset": "abs(SA return - baseline return)>1e-8; positive AND negative records retained",
                              "different_actions": "unequal complete recorded action sequences for the paired episode, including unequal lengths",
                              "initial_behavior_cohort": "post-hoc diagnostic cohort defined by action disagreement at checkpoint zero, then kept fixed",
                              "reward_curve_groups": "only exact equality of every per-unit return, for every seed and checkpoint; no rounding or curve shifts"},
              "limitations": ["Outcome-conditioned subsets are post-hoc diagnostics; they cannot establish general superiority.",
                              "Dynamic subset composition changes with training; conditional curves do not isolate a learning effect.",
                              "Pooled records repeat windows across seeds and workflows; no independent-cluster or significance claim.",
                              "Trajectory disagreement is not fixed-state counterfactual replay or proof of beneficial action diversity.",
                              "No cumulative differences, positive-only filtering, padding, fitted trends or reward modification.",
                              "Monotonicity of sampled reward gaps and a reward plateau do not establish policy convergence.",
                              "New long training and mechanism opportunity repairs are not performed by this plotting tool."],
              "source_files_sha256": source_hashes, "script_sha256": sha(Path(__file__)),
              "output_files_sha256": {p.name: sha(p) for p in out.iterdir()}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), "checks": report["checks"], "groups": groups,
                      "final": report["final_decomposition"], "trends": trends}, ensure_ascii=False))


def render(out, plan, points, curves, groups, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    x = np.asarray(points) / 1000

    def finish(fig, name):
        fig.savefig(out / f"{name}.png", dpi=190, facecolor="white")
        fig.savefig(out / f"{name}.pdf", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    styles = ["-", "--", "-.", ":"]
    for i, group in enumerate(groups):
        label = " / ".join(LABELS[n] for n in group)
        for ax in axes:
            ax.plot(x, curves[group[0]], linestyle=styles[i % 4], marker="o", markersize=2,
                    color=plt.get_cmap("tab10")(i), lw=1.6, label=label)
    axes[0].set(title="All checkpoints: full paired population", ylabel="Native validation episode reward")
    axes[1].set(title="Late-stage detail (explicit axis zoom)", xlim=(points[len(points)//2]/1000, x[-1]), ylim=(17.9, 19.1))
    for ax in axes:
        ax.set_xlabel("Training environment interactions (thousands)")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, .035))
    fig.suptitle(f"{len(plan['agents'])} algorithms, {len(groups)} distinct complete reward histories | duplicate histories grouped")
    fig.text(.5, .012, "All algorithm names retained. Grouping requires every per-unit return to match at every checkpoint and seed. No smoothing or curve displacement.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .19, 1, .94))
    finish(fig, "grouped_full_reward_curves")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, baseline in zip(axes.flat, [n for n in plan["agents"] if n != "sa_ghmappo"]):
        selected = [r for r in rows if r["baseline"] == baseline and r["seed"] == "pooled"]
        for seed in plan["seeds"]:
            seed_rows = [r for r in rows if r["baseline"] == baseline and r["seed"] == str(seed)]
            ax.plot(x, [r["full_gap"] for r in seed_rows], color="#176ab5", lw=.6, alpha=.25)
        ax.plot(x, [r["full_gap"] for r in selected], color="#176ab5", lw=1.8, label="Full-population gap")
        ax.plot(x, [np.nan if r["nonzero_reward_conditional_gap"] is None else r["nonzero_reward_conditional_gap"] for r in selected],
                color="#a044ab", lw=1.6, marker=".", markersize=3, label="Nonzero reward differences only")
        ax.plot(x, [np.nan if r["initial_behavior_cohort_gap"] is None else r["initial_behavior_cohort_gap"] for r in selected],
                color="#dc8c20", lw=1.2, linestyle="--", label="Fixed initial-action-disagreement cohort")
        ax.axhline(0, color="#333333", lw=.65, linestyle=":")
        final = selected[-1]
        ax.set_title(f"SA-GHMAPPO minus {LABELS[baseline]}\nFinal nonzero-return coverage: {final['different_reward_units']}/{final['paired_units']} ({final['different_reward_fraction']:.1%})", fontsize=10)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Training interactions (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Signed native reward difference")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .045))
    fig.suptitle("Difference-only views: every baseline and all signs retained | diagnostic, not a superiority test", fontsize=14)
    fig.text(.5, .015, "Purple conditions on observed reward differences; changing membership can amplify apparent gaps. Blue retains the complete denominator.\nOrange keeps a fixed cohort selected from initial action traces. Thin blue: each seed. All 33 recorded checkpoints; no smoothing or cumulative sums.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .95))
    finish(fig, "signed_difference_parts")

    final_rows = [r for r in rows if r["seed"] == "pooled" and r["training_steps"] == points[-1]]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(final_rows))
    ax.barh(y + .17, [r["different_action_fraction"] * 100 for r in final_rows], height=.32,
            color="#176ab5", label="Different complete action trajectories")
    ax.barh(y - .17, [r["different_reward_fraction"] * 100 for r in final_rows], height=.32,
            color="#a044ab", label="Different episode rewards (either sign)")
    ax.set_yticks(y, [LABELS[r["baseline"]] for r in final_rows])
    ax.invert_yaxis()
    ax.set(xlim=(0, 108), xlabel="Fraction of paired episode records (%)",
           title="Action diversity and reward differences are distinct\nFinal checkpoint: SA-GHMAPPO versus each baseline; all 72 paired records retained")
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    for i, row in enumerate(final_rows):
        for offset, key in [(.17, "different_action_fraction"), (-.17, "different_reward_fraction")]:
            value = row[key] * 100
            ax.text(value + 1, i + offset, f"{value:.1f}%", va="center", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(.5, .045))
    fig.text(.5, .012, "Descriptive paired records repeat windows across seeds; no independent-sample or significance claim. Reward differences include losses.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    finish(fig, "action_vs_reward_difference_coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    export(args.run_root.resolve(), args.audit_report.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
