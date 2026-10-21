"""Matched multi-cycle development training; isolated from algorithm optimization.

Reuses the frozen native reward/environment/configuration path of the short
experiment, with a predeclared larger common interaction/update budget.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import run_fair_reward_training as fair
from diagnose_sagh_flat_reward import action_record, compare_records, observe


def check_plan(p):
    cycles = p["training_cycles"]
    if not isinstance(cycles, int) or cycles < 2:
        raise ValueError("Multiple complete training cycles required")
    one = dict(p, training_steps=p["training_steps"] // cycles)
    fair.check_plan(one)
    if one["training_steps"] * cycles != p["training_steps"]:
        raise ValueError("Incomplete cycle budget")
    updates = p["training_steps"] // p["steps_per_update"]
    if updates % p["evaluate_every_updates"]:
        raise ValueError("Final validation must be scheduled")
    if p["plotting"] != {"episode_trailing_mean": 100, "retain_all_raw_episodes": True,
                          "validation_smoothing": False, "curve_shift_or_padding": False}:
        raise ValueError("Frozen plot transparency rules changed")


def append_rows(path, rows):
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def prepare(source, root, config, manifest):
    if root.exists() or root.with_name(root.name + ".preparing").exists():
        raise FileExistsError("Preparation is create-only")
    p = json.loads(config.read_text())
    check_plan(p)
    temporary = root.with_name(root.name + ".preparing")
    original_check = fair.check_plan
    # The base preparer freezes actual configs, data and frame identities.
    fair.check_plan = lambda value: check_plan_with_base(value, original_check)
    try:
        fair.prepare(source, temporary, config, manifest)
    finally:
        fair.check_plan = original_check
    p = json.loads((temporary / "frozen_plan.json").read_text())
    p.pop("plan_sha256")
    p["base_runner_sha256"] = p.pop("runner_sha256")
    p["runner_sha256"] = fair.sha(__file__)
    p["diagnostic_observer_sha256"] = fair.sha(Path(__file__).with_name("diagnose_sagh_flat_reward.py"))
    from src.evaluators.main_results_support import (classify_window_activity, _estimate_prediction_activity,
                                                   _estimate_physical_transfer_activity)
    from src.evaluators.real_sample_support import scan_mobility_windows
    for window in p["windows"]:
        bundle = fair.window_bundle(source, p, window["frame_offset"])
        scans = scan_mobility_windows(frames=bundle.frames, layout_candidates=[p["rsu_layout"]],
                    frame_offset=0, window_length=p["window_length"], stride=p["window_length"], ranking_mode="max_handoff_candidate")
        if len(scans) != 1:
            raise ValueError("Exact frozen window must yield one classification record")
        scanned = scans[0]
        predicted = _estimate_prediction_activity(bundle.frames, bundle.rsu_states)
        physical = _estimate_physical_transfer_activity(bundle.frames, bundle.rsu_states)
        activity = {"mechanism_signal_count": int(physical["physical_transfer_opportunity_count"] if physical["physical_transfer_activity_available"] else scanned["estimated_handoff_count"]),
                    "active_vehicle_count_mean": float(scanned["active_vehicle_count_mean"]),
                    "predicted_next_ratio": float(predicted["predicted_next_rsu_non_null_ratio"]),
                    "predicted_handoff_ratio": float(predicted["predicted_handoff_target_non_null_ratio"]),
                    "estimated_association_change_count": int(scanned["estimated_association_change_count"])}
        window["window_class"] = classify_window_activity(**activity)
        window["window_class_activity"] = activity
    p["window_class_protocol"] = "native_data_only_thresholds_v1_no_reward_selection"
    p["probability_contract"] = "sampled_base_and_option_v1"
    p["supersedes_user_paused_run"] = "long_fair_reward_training_20261021_v1"
    p["parent_experiment"] = "fair_reward_training_20261021_v2"
    p["parent_plan_sha256"] = "81de66f5f104ebcf582828a0dda941cf99079521eec026c434601bb21d13053b"
    p["protected_files"] = ["scripts/train_sa_ghmappo_real_sample.py", "src/agents/sa_ghmappo_agent.py",
                            "src/agents/sa_ghmappo_core.py", "src/encoders/fusion_encoder.py",
                            "src/evaluators/real_eval_support.py", "tests/test_algo_pool_contract.py",
                            "tests/test_checkpoint_compat.py"]
    p["protected_diff_sha256"] = "1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2"
    p["plan_sha256"] = fair.semantic_hash(p)
    fair.write_json(temporary / "frozen_plan.json", p)
    fair.write_json(temporary / "status.json", {"status": "PREPARED", "plan_sha256": p["plan_sha256"]})
    temporary.rename(root)
    print(json.dumps({"status": "LONG_PLAN_FROZEN", "plan_sha256": p["plan_sha256"],
                      "steps_per_cell": p["training_steps"], "cycles": p["training_cycles"]}), flush=True)


def check_plan_with_base(p, base_check):
    # Avoid recursive replacement while the reusable base preparer calls us.
    current = fair.check_plan
    fair.check_plan = base_check
    try:
        check_plan(p)
    finally:
        fair.check_plan = current


def evaluate(source, root, p, name, seed, agent, bundles, workflows, step, cell, references):
    old_rng = (random.getstate(), fair.np.random.get_state(), fair.torch.random.get_rng_state())
    results, probes = [], []
    try:
        eval_agent = fair.fresh_agent(name, p["agent_configs"][name][str(seed)], seed)
        eval_agent._network.load_state_dict(agent._network.state_dict())
        eval_agent._update_count = agent._update_count
        eval_agent._deterministic_action = True
        eval_agent._network.eval()
        if name == "sa_ghmappo":
            observe(eval_agent)
        with (cell / f"validation_actions_{step:07d}.jsonl").open("x") as trace:
            for i, bundle in enumerate(bundles):
                fair.yield_resources(root)
                for j, workflow in enumerate(workflows):
                    metadata = {"phase": "validation", "training_steps": step,
                                "window_id": bundle.rsu_metadata["window_id"], "workflow_id": workflow.workflow_id,
                                "window_class": bundle.rsu_metadata["window_class"]}
                    summary, rollout = fair.make_episode(source, p, eval_agent, bundle, workflow,
                                                         seed+10000+i*len(workflows)+j,
                                                         p["max_episode_steps"], metadata, False)
                    metrics = summary["system_metrics"]
                    results.append({"agent": name, "seed": seed, "training_steps": step,
                                    "window_id": metadata["window_id"], "workflow_id": workflow.workflow_id,
                                    "total_reward": summary["reward_breakdown"]["total"]["sum"],
                                    "completed": bool(summary["episode_status"].get("completed", False)),
                                    "window_class": metadata["window_class"],
                                    "workflow_continuity_rate": metrics.get("workflow_continuity_rate"),
                                    "backhaul_traffic_cost": metrics.get("backhaul_traffic_cost")})
                    for k, row in enumerate(rollout):
                        record = action_record(row) if name == "sa_ghmappo" else {
                            "action": int(row["action"]), "reward": float(row["reward"]),
                            "aggregation_reason": row["action_info"].get("aggregation_reason")}
                        json_line(trace, {"window_id": metadata["window_id"], "workflow_id": workflow.workflow_id,
                                          "step": k, "window_class": metadata["window_class"], **record})
                        if name == "sa_ghmappo" and step == 0:
                            references.append((copy.deepcopy(row["observation"]), copy.deepcopy(row["decision_info"]), record))
        if name == "sa_ghmappo":
            for k, (observation, info, initial) in enumerate(references):
                info = copy.deepcopy(info)
                info["run_metadata"]["training_steps"] = step
                action, diagnostics = eval_agent.act(observation, info)
                current = action_record({"action": action, "reward": 0., "action_info": diagnostics})
                probes.append({"training_steps": step, "reference_state": k, **compare_records(initial, current)})
            append_rows(cell / "paired_initial_states.csv", probes)
    finally:
        random.setstate(old_rng[0])
        fair.np.random.set_state(old_rng[1])
        fair.torch.random.set_rng_state(old_rng[2])
    fair.csv_write(cell / f"evaluation_{step:07d}.csv", results)
    append_rows(cell / "evaluation.csv", results)


def run_cell(source, root, p, name, seed, train, validation, workflows):
    cell = root / "cells" / name / f"seed{seed}"
    if cell.exists():
        raise FileExistsError(f"Refusing restart or overwrite: {cell}")
    cell.mkdir(parents=True)
    agent = fair.fresh_agent(name, p["agent_configs"][name][str(seed)], seed)
    agent.save(str(cell / "initial.pt"))
    fair.write_json(cell / "identity.json", {"agent": name, "seed": seed, "plan_sha256": p["plan_sha256"],
                    "fresh_initialization": True, "actual_config": agent._checkpoint_config(),
                    "replay_initial_size": len(getattr(agent, "_replay", []))})
    references = []
    evaluate(source, root, p, name, seed, agent, validation, workflows, 0, cell, references)
    pending, history = [], []
    total_steps, episode, update = 0, 0, 0
    units = [(i, j) for j in range(len(workflows)) for i in range(len(train))]
    started = time.time()
    with (cell / "episode_receipts.jsonl").open("x") as receipts, (cell / "updates.jsonl").open("x") as learning:
        for cycle in range(p["training_cycles"]):
            for local_unit, (i, j) in enumerate(units):
                remaining = p["steps_per_training_unit"]
                while remaining:
                    fair.yield_resources(root)
                    episode += 1
                    max_steps = min(p["max_episode_steps"], remaining)
                    metadata = {"phase": "train", "agent": name, "seed": seed, "episode_index": episode,
                                "cycle": cycle+1, "unit_index": cycle*len(units)+local_unit,
                                "window_id": train[i].rsu_metadata["window_id"], "workflow_id": workflows[j].workflow_id,
                                "window_class": train[i].rsu_metadata["window_class"],
                                "plan_sha256": p["plan_sha256"]}
                    summary, rollout = fair.make_episode(source, p, agent, train[i], workflows[j], seed+local_unit,
                                                         max_steps, metadata, True)
                    for row in rollout:
                        if row["decision_info"]["run_metadata"].get("window_class") != metadata["window_class"]:
                            raise ValueError("Window classification did not reach learner")
                        if hasattr(agent, "_latent_policy_statistics") and row["action_info"].get("probability_contract") != p["probability_contract"]:
                            raise ValueError("Unverified action probability contract")
                    count = len(rollout)
                    if not 0 < count <= remaining:
                        raise ValueError("Interaction accounting failure")
                    remaining -= count
                    total_steps += count
                    pending.extend(rollout)
                    history.append({"agent": name, "seed": seed, "episode_index": episode, "training_steps": total_steps,
                                    "cycle": cycle+1, "unit_index": metadata["unit_index"],
                                    "window_class": metadata["window_class"],
                                    "window_id": metadata["window_id"], "workflow_id": metadata["workflow_id"],
                                    "episode_steps": count, "budget_cut": count == max_steps and max_steps < p["max_episode_steps"],
                                    "total_reward": summary["reward_breakdown"]["total"]["sum"],
                                    "completed": bool(summary["episode_status"].get("completed", False)),
                                    "action_counts": json.dumps(dict(Counter(int(r["action"]) for r in rollout)), sort_keys=True)})
                    behaviors = []
                    for row in rollout:
                        action_info = row["action_info"]
                        behavior = {key: action_info.get(key) for key in ("probability_contract", "sampled_base_action", "sampled_base_log_prob", "log_prob", "sampled_base_probs", "head_actions", "env_action_probability_scope")}
                        option = action_info.get("option_gate", {})
                        behavior["option_gate"] = {key: option[key] for key in ("enabled", "option_action", "option_log_prob", "option_probs", "option_mask", "prior_target", "option_actions", "base_env_action") if key in option}
                        behaviors.append(behavior)
                    json_line(receipts, {"metadata": metadata, "training_steps": total_steps,
                                         "behavior_diagnostics": behaviors,
                                         "episode_status": summary["episode_status"], "reward_breakdown": summary["reward_breakdown"],
                                         "system_metrics": summary["system_metrics"], "actions": [int(r["action"]) for r in rollout]})
                if len(pending) == p["steps_per_update"]:
                    fair.yield_resources(root)
                    before = time.time()
                    info = agent.learn(pending)
                    pending = []
                    update += 1
                    json_line(learning, {"update": update, "training_steps": total_steps, "cycle": cycle+1,
                                          "learning_seconds": time.time()-before, "learn_info": info})
                    append_rows(cell / "training.csv", history)
                    history = []
                    if update % p["evaluate_every_updates"] == 0:
                        agent.save(str(cell / f"update_{update:04d}.pt"))
                        evaluate(source, root, p, name, seed, agent, validation, workflows, total_steps, cell, references)
                    fair.write_json(root / "status.json", {"status": "RUNNING", "agent": name, "seed": seed,
                                    "cell_training_steps": total_steps, "cell_updates": update, "cycle": cycle+1,
                                    "cell_elapsed_seconds": time.time()-started, "plan_sha256": p["plan_sha256"]})
                    print(f"{name} seed={seed} cycle={cycle+1}/{p['training_cycles']} steps={total_steps}/{p['training_steps']} update={update}", flush=True)
                elif len(pending) > p["steps_per_update"]:
                    raise ValueError("Update budget exceeded")
    if total_steps != p["training_steps"] or pending or history or update != p["training_steps"] // p["steps_per_update"]:
        raise ValueError("Final training budget mismatch")
    fair.write_json(cell / "complete.json", {"status": "COMPLETED", "training_steps": total_steps,
                    "updates": update, "cycles": p["training_cycles"], "episodes": episode, "wall_seconds": time.time()-started,
                    "plan_sha256": p["plan_sha256"], "training_csv_sha256": fair.sha(cell / "training.csv"),
                    "evaluation_csv_sha256": fair.sha(cell / "evaluation.csv"),
                    "episode_receipts_sha256": fair.sha(cell / "episode_receipts.jsonl"),
                    "updates_sha256": fair.sha(cell / "updates.jsonl")})


def execute(source, root):
    p = json.loads((root / "frozen_plan.json").read_text())
    expected = p.pop("plan_sha256")
    if fair.semantic_hash(p) != expected or fair.sha(__file__) != p["runner_sha256"] or fair.sha(fair.__file__) != p["base_runner_sha256"]:
        raise ValueError("Frozen runner/plan changed")
    if fair.sha(Path(__file__).with_name("diagnose_sagh_flat_reward.py")) != p["diagnostic_observer_sha256"]:
        raise ValueError("Observer changed")
    p["plan_sha256"] = expected
    check_plan(p)
    if str(source) != p["source_root"] or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != p["source_commit"]:
        raise ValueError("Pinned clean source changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], cwd=source, check=True)
    import hashlib
    protected = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *p["protected_files"]], cwd=fair.PROJECT)
    if hashlib.sha256(protected).hexdigest() != p["protected_diff_sha256"]:
        raise ValueError("Protected mainline modifications changed before launch")
    for key, identity in p["data_files"].items():
        if fair.sha(p[key]) != identity["sha256"]:
            raise ValueError(f"Raw data changed: {key}")
    bundles = {"train": [], "validation": []}
    for w in p["windows"]:
        bundle = fair.window_bundle(source, p, w["frame_offset"])
        identity = fair.window_identity(bundle)
        frames_hash = fair.semantic_hash([{**{k: v for k, v in f.items() if k != "vehicles"},
                                          "vehicles": [v.to_dict() for v in f["vehicles"]]} for f in bundle.frames])
        if identity != {k: w[k] for k in identity} or frames_hash != w["frame_content_sha256"]:
            raise ValueError("Actual window identity/content changed")
        if w.get("window_class") not in {"mechanism_activating", "active_non_mechanism", "idle_or_sparse"}:
            raise ValueError("Missing frozen window class")
        bundle.rsu_metadata["window_class"] = w["window_class"]
        bundles[w["split"]].append(bundle)
    fair.intervals_disjoint(p["windows"])
    workflows = fair.build_workflows(workflow_csv_path=p["workflow_path"], max_workflows=p["max_workflows"],
                                    workflow_selector="ordered", min_tasks=5, max_tasks=20, random_seed=7)
    if fair.semantic_hash([w.to_dict() for w in workflows]) != p["workflow_content_sha256"]:
        raise ValueError("Workflow changed")
    for seed in p["seeds"]:
        for name in p["agents"]:
            fair.yield_resources(root)
            run_cell(source, root, p, name, seed, bundles["train"], bundles["validation"], workflows)
    fair.write_json(root / "status.json", {"status": "TRAINING_COMPLETED", "cells": len(p["agents"])*len(p["seeds"]), "plan_sha256": expected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=fair.PROJECT / "configs/experiment/long_fair_reward_training_v1.json")
    parser.add_argument("--manifest", type=Path, default=fair.PROJECT / "artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/seed_checkpoint_manifest.json")
    args = parser.parse_args()
    os.nice(max(0, 19-os.nice(0)))
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    fair.load_api(args.source_root.resolve())
    if args.mode == "prepare":
        prepare(args.source_root.resolve(), args.output_root.resolve(), args.config.resolve(), args.manifest.resolve())
    else:
        execute(args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
