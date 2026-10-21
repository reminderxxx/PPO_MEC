"""Independent, fixed-budget reward learning curves; never edits algorithm code.

The checkpoint manifest supplies *configuration only*. All weights, optimizer,
replay and schedules start fresh. Each algorithm sees identical per-unit step
budgets. A pinned clean source checkout isolates concurrent optimization work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_plan(p):
    if p["training_steps"] % p["steps_per_update"] or p["steps_per_update"] % p["steps_per_training_unit"]:
        raise ValueError("Training/update/unit budgets must divide exactly")
    if p["training_steps"] != p["train_window_count"] * p["max_workflows"] * p["steps_per_training_unit"]:
        raise ValueError("All training units must receive exactly one fixed step allocation")
    if p["resources"] != {"device": "cpu", "torch_threads": 1, "concurrent_cells": 1, "nice": 19, "yield_to_foreign_training": True}:
        raise ValueError("Resource isolation must remain enabled")
    if p["external_reward_shaping"] or p["initialization"] != "fresh_weights_and_optimizer_no_checkpoint_load":
        raise ValueError("Shared native reward and fresh initialization are required")


def intervals_disjoint(windows):
    for i, a in enumerate(windows):
        for b in windows[i + 1:]:
            if a["source_segment_id"] == b["source_segment_id"]:
                if max(a["time_index_start"], b["time_index_start"]) <= min(a["time_index_end"], b["time_index_end"]):
                    raise ValueError(f"Overlapping source intervals: {a['window_id']} / {b['window_id']}")


def foreign_training(process_text, own_pid):
    """Conservative process guard; ignore shells whose arguments quote scripts."""
    result = []
    for line in process_text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, comm, args = fields
        if int(pid) == own_pid or "python" not in Path(comm).name.lower():
            continue
        tokens = shlex.split(args)
        scripts = [Path(t).name.lower() for t in tokens[1:] if t.endswith(".py")]
        if any("train" in s or "benchmark" in s or "closed_loop" in s for s in scripts):
            result.append({"pid": int(pid), "command": args})
    return result


def yield_resources(root):
    while True:
        process_text = subprocess.check_output(["ps", "-axo", "pid=,comm=,args="], text=True)
        others = foreign_training(process_text, os.getpid())
        if not others:
            return
        write_json(root / "resource_wait.json", {"status": "YIELDING_TO_OTHER_TRAINING", "observed_at": time.time(), "processes": others})
        time.sleep(5)


def load_api(source):
    sys.path.insert(0, str(source))
    global np, torch, build_agent, filter_config, load_bundle, build_workflows
    global clone_frames, clone_rsu_state, clone_workflow_state, ReplayProvider
    global VecWorkflowCoreEnv, PredictorManager, EpisodeRecorder, GymVecEnv, Trainer, AdapterCatalog
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from src.agents.registry import build_agent
    from src.evaluators.real_eval_support import _filter_checkpoint_config as filter_config
    from src.evaluators.main_results_support import (
        load_window_bundle as load_bundle, build_selected_workflow_states as build_workflows,
        clone_frames, clone_rsu_state, clone_workflow_state,
    )
    from src.data.mobility.replay_provider import ReplayProvider
    from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
    from src.envs.core.predictor_manager import PredictorManager
    from src.metrics.recorder import EpisodeRecorder
    from src.envs.wrappers.gym_vec_env import GymVecEnv
    from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer as Trainer
    from src.data.model_catalog.adapter_catalog import AdapterCatalog


def window_bundle(source, p, offset):
    return load_bundle(root_dir=source, mobility_source="ngsim", mobility_csv_path=p["mobility_path"],
                       max_mobility_rows=p["max_mobility_rows"], rsu_layout=p["rsu_layout"],
                       frame_offset=offset, window_length=p["window_length"], random_seed=7)


def window_identity(bundle):
    keys = ("window_id", "frame_offset", "window_length", "time_index_start", "time_index_end", "source_segment_id")
    return {k: bundle.rsu_metadata[k] for k in keys}


def fresh_agent(name, config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = filter_config(name, config)
    if name == "sa_ghmappo":
        # The inference loader intentionally omits training-only settings.
        # Preserve them here; structural aliases are handled by that loader.
        structural = {"agent_name", "policy_type", "encoder_kind", "centralized_critic",
                      "hierarchical_conditioning", "event_head_enabled", "adapter_prefetch_enabled"}
        kwargs.update({k: v for k, v in config.items() if k not in structural})
    kwargs.update(random_seed=seed, deterministic_action=False, device="cpu")
    agent = build_agent(name, **kwargs)
    current = agent._checkpoint_config()
    differences = {k: {"source": v, "actual": current.get(k)} for k, v in config.items()
                   if k in current and json.loads(json.dumps(v)) != json.loads(json.dumps(current[k]))}
    if differences:
        raise ValueError(f"Configuration changed for {name}: {differences}")
    if agent._update_count != 0 or len(agent._optimizer.state) != 0:
        raise ValueError("Fresh initialization failed")
    return agent


def prepare(source, root, config, manifest):
    if root.exists():
        raise FileExistsError("Preparation is create-only")
    p = json.loads(config.read_text())
    check_plan(p)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], cwd=source, check=True, stdout=subprocess.DEVNULL)
    p.update(source_root=str(source), source_commit=source_commit, project_root=str(PROJECT),
             mobility_path=str(next((PROJECT / "data/raw/mobility/ngsim").glob("*.csv"))),
             workflow_path=str(PROJECT / "data/raw/workflow/alibaba2018/batch_task.csv"),
             config_sha256=sha(config), manifest_path=str(manifest), manifest_sha256=sha(manifest),
             runner_sha256=sha(__file__), prepared_at=time.time())
    m = json.loads(manifest.read_text())
    sources = []
    configs = {}
    for name in p["agents"]:
        configs[name] = {}
        for seed in p["seeds"]:
            checkpoint = Path(m[name][str(seed)])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg = payload["config"]
            fresh_agent(name, cfg, seed)
            configs[name][str(seed)] = cfg
            sources.append({"agent": name, "seed": seed, "configuration_source_path": str(checkpoint),
                            "source_checkpoint_sha256": sha(checkpoint), "weights_loaded": False, "optimizer_loaded": False})
    windows = []
    for split, offsets in (("train", [i * p["window_stride"] for i in range(p["train_window_count"])]),
                           ("validation", [p["validation_first_offset"] + i * p["window_stride"] for i in range(p["validation_window_count"])])):
        for offset in offsets:
            bundle = window_bundle(source, p, offset)
            identity = window_identity(bundle)
            if identity["frame_offset"] != offset or identity["window_length"] != p["window_length"]:
                raise ValueError("Loader silently shifted or shortened window")
            identity["split"] = split
            identity["frame_content_sha256"] = semantic_hash([
                {**{k: v for k, v in f.items() if k != "vehicles"}, "vehicles": [v.to_dict() for v in f["vehicles"]]} for f in bundle.frames
            ])
            windows.append(identity)
    intervals_disjoint(windows)
    workflows = build_workflows(workflow_csv_path=p["workflow_path"], max_workflows=p["max_workflows"],
                                workflow_selector="ordered", min_tasks=5, max_tasks=20, random_seed=7)
    p["workflow_ids"] = [w.workflow_id for w in workflows]
    p["workflow_content_sha256"] = semantic_hash([w.to_dict() for w in workflows])
    p["data_files"] = {key: {"size": Path(p[key]).stat().st_size, "sha256": sha(p[key])} for key in ("mobility_path", "workflow_path")}
    p["agent_configs"] = configs
    p["windows"] = windows
    p["configuration_sources"] = sources
    p["plan_sha256"] = semantic_hash(p)
    root.mkdir(parents=True)
    write_json(root / "frozen_plan.json", p)
    write_json(root / "status.json", {"status": "PREPARED", "training_steps": 0, "plan_sha256": p["plan_sha256"]})
    print(json.dumps({"status": "PREPARED", "plan_sha256": p["plan_sha256"], "windows": len(windows), "cells": len(sources)}))


def make_episode(source, p, agent, bundle, workflow, seed, max_steps, metadata, training):
    recorder = EpisodeRecorder(prefetch_validation_window=6)
    catalog = AdapterCatalog.from_json(source / "src/data/model_catalog/sample_model_catalog.json")
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=clone_frames(bundle.frames)),
        workflow_state=clone_workflow_state(workflow), adapter_catalog=catalog,
        rsu_states=[clone_rsu_state(r) for r in bundle.rsu_states],
        predictor_manager=PredictorManager(horizon=p["prediction_horizon"], random_seed=seed),
        max_steps=max(p["max_episode_steps"] + 2, 8), mobility_source="ngsim",
        primary_vehicle_selection=p["primary_vehicle_selection"], reward_positive_offset=p["reward_positive_offset"],
    )
    env = GymVecEnv(core_env=core, recorder=recorder)
    trainer = Trainer(env=env, agent=agent, recorder=recorder, max_steps=max_steps, gamma=.99, gae_lambda=.95)
    summary, rollout = trainer.collect_episode(run_metadata=metadata, collect_model_targets=training)
    if training:
        if agent.agent_name == "sa_ghmappo":
            from scripts.train_sa_ghmappo_real_sample import annotate_retrospective_handoff_auxiliary_targets
            summary["retrospective_handoff_auxiliary"] = annotate_retrospective_handoff_auxiliary_targets(
                rollout=rollout, env=env, agent=agent)
        for row in rollout:
            if not math.isfinite(float(row["reward"])):
                raise ValueError("Non-finite native reward")
        reward = sum(float(row["reward"]) for row in rollout)
        if not math.isclose(reward, summary["reward_breakdown"]["total"]["sum"], abs_tol=1e-5):
            raise ValueError("Learning rewards differ from native episode reward")
    return summary, rollout


def evaluate(source, root, p, name, seed, agent, bundles, workflows, step, cell):
    # Isolate evaluation runtime/cache/RNG state from the training agent.
    # Rebuild the same frozen architecture; copy only its current network state.
    old_rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    results = []
    try:
        eval_agent = fresh_agent(name, p["agent_configs"][name][str(seed)], seed)
        eval_agent._network.load_state_dict(agent._network.state_dict())
        eval_agent._update_count = agent._update_count
        eval_agent._deterministic_action = True
        eval_agent._network.eval()
        for i, bundle in enumerate(bundles):
            yield_resources(root)
            for j, workflow in enumerate(workflows):
                summary, _ = make_episode(source, p, eval_agent, bundle, workflow, seed + 10000 + i * 2 + j,
                                          p["max_episode_steps"], {"phase": "validation", "training_steps": step,
                                          "window_id": bundle.rsu_metadata["window_id"], "workflow_id": workflow.workflow_id}, False)
                metrics = summary["system_metrics"]
                results.append({"agent": name, "seed": seed, "training_steps": step,
                                "window_id": bundle.rsu_metadata["window_id"], "workflow_id": workflow.workflow_id,
                                "total_reward": summary["reward_breakdown"]["total"]["sum"],
                                "completed": bool(summary["episode_status"].get("completed", False)),
                                "workflow_continuity_rate": metrics.get("workflow_continuity_rate"),
                                "backhaul_traffic_cost": metrics.get("backhaul_traffic_cost")})
    finally:
        random.setstate(old_rng[0]); np.random.set_state(old_rng[1]); torch.random.set_rng_state(old_rng[2])
    csv_write(cell / f"evaluation_{step:07d}.csv", results)
    return results


def run_cell(source, root, p, name, seed, train, validation, workflows):
    cell = root / "cells" / name / f"seed{seed}"
    if cell.exists():
        raise FileExistsError(f"Refusing to restart/overwrite cell {cell}")
    cell.mkdir(parents=True)
    agent = fresh_agent(name, p["agent_configs"][name][str(seed)], seed)
    agent.save(str(cell / "initial.pt"))
    write_json(cell / "identity.json", {"agent": name, "seed": seed, "plan_sha256": p["plan_sha256"],
               "fresh_initialization": True, "parameter_count": sum(v.numel() for v in agent._network.parameters()),
               "actual_config": agent._checkpoint_config(), "replay_initial_size": len(getattr(agent, "_replay", []))})
    evaluations = evaluate(source, root, p, name, seed, agent, validation, workflows, 0, cell)
    history, pending, updates = [], [], []
    total_steps, episode = 0, 0
    # Each workflow visits every train window; step exposure is matched exactly.
    units = [(i, j) for j in range(len(workflows)) for i in range(len(train))]
    started = time.time()
    for unit_index, (i, j) in enumerate(units):
        remaining = p["steps_per_training_unit"]
        while remaining:
            yield_resources(root)
            episode += 1
            max_steps = min(p["max_episode_steps"], remaining)
            metadata = {"phase": "train", "agent": name, "seed": seed, "episode_index": episode,
                        "unit_index": unit_index, "window_id": train[i].rsu_metadata["window_id"],
                        "workflow_id": workflows[j].workflow_id, "plan_sha256": p["plan_sha256"]}
            summary, rollout = make_episode(source, p, agent, train[i], workflows[j], seed + unit_index,
                                            max_steps, metadata, True)
            count = len(rollout)
            if not 0 < count <= remaining:
                raise ValueError("Interaction accounting failure")
            remaining -= count
            total_steps += count
            pending.extend(rollout)
            history.append({"agent": name, "seed": seed, "episode_index": episode, "training_steps": total_steps,
                            "window_id": train[i].rsu_metadata["window_id"], "workflow_id": workflows[j].workflow_id,
                            "episode_steps": count, "budget_cut": count == max_steps and max_steps < p["max_episode_steps"],
                            "total_reward": summary["reward_breakdown"]["total"]["sum"],
                            "completed": bool(summary["episode_status"].get("completed", False))})
            # Preserve compact raw native receipts, without copying large traces.
            write_json(cell / f"episode_{episode:05d}.json", {"run_info": summary["run_info"], "episode_status": summary["episode_status"],
                       "reward_breakdown": summary["reward_breakdown"], "system_metrics": summary["system_metrics"], "metadata": metadata})
        if len(pending) == p["steps_per_update"]:
            yield_resources(root)
            learning_started = time.time()
            info = agent.learn(pending)
            pending = []
            update = total_steps // p["steps_per_update"]
            updates.append({"update": update, "training_steps": total_steps, "learning_seconds": time.time() - learning_started, "learn_info": info})
            checkpoint = cell / f"update_{update:04d}.pt"
            agent.save(str(checkpoint))
            if update % p["evaluate_every_updates"] == 0:
                evaluations.extend(evaluate(source, root, p, name, seed, agent, validation, workflows, total_steps, cell))
            csv_write(cell / "training.csv", history)
            csv_write(cell / "evaluation.csv", evaluations)
            write_json(cell / "updates.json", updates)
            write_json(root / "status.json", {"status": "RUNNING", "agent": name, "seed": seed,
                       "cell_training_steps": total_steps, "cell_updates": update, "cell_elapsed_seconds": time.time() - started})
            print(f"{name} seed={seed} steps={total_steps}/{p['training_steps']} update={update} elapsed={time.time()-started:.1f}s", flush=True)
        elif len(pending) > p["steps_per_update"]:
            raise ValueError("Update budget exceeded")
    if total_steps != p["training_steps"] or pending or len(updates) != p["training_steps"] // p["steps_per_update"]:
        raise ValueError("Final training budget mismatch")
    write_json(cell / "complete.json", {"status": "COMPLETED", "training_steps": total_steps,
               "updates": len(updates), "episodes": episode, "wall_seconds": time.time() - started,
               "plan_sha256": p["plan_sha256"], "training_csv_sha256": sha(cell / "training.csv"),
               "evaluation_csv_sha256": sha(cell / "evaluation.csv")})


def execute(source, root):
    p = json.loads((root / "frozen_plan.json").read_text())
    expected = p.pop("plan_sha256")
    if semantic_hash(p) != expected or sha(__file__) != p["runner_sha256"]:
        raise ValueError("Frozen plan or runner changed")
    p["plan_sha256"] = expected
    check_plan(p)
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if str(source) != p["source_root"] or actual_commit != p["source_commit"]:
        raise ValueError("Execution source differs from pinned plan")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], cwd=source, check=True, stdout=subprocess.DEVNULL)
    for key, identity in p["data_files"].items():
        if sha(p[key]) != identity["sha256"]:
            raise ValueError(f"Data source changed: {key}")
    bundles = {"train": [], "validation": []}
    for w in p["windows"]:
        b = window_bundle(source, p, w["frame_offset"])
        if window_identity(b) != {k: w[k] for k in window_identity(b)}:
            raise ValueError("Window identity changed before rollout")
        bundles[w["split"]].append(b)
    workflows = build_workflows(workflow_csv_path=p["workflow_path"], max_workflows=p["max_workflows"],
                                workflow_selector="ordered", min_tasks=5, max_tasks=20, random_seed=7)
    if semantic_hash([w.to_dict() for w in workflows]) != p["workflow_content_sha256"]:
        raise ValueError("Workflow identity changed")
    for seed in p["seeds"]:
        for name in p["agents"]:
            run_cell(source, root, p, name, seed, bundles["train"], bundles["validation"], workflows)
    write_json(root / "status.json", {"status": "TRAINING_COMPLETED", "cells": len(p["agents"]) * len(p["seeds"]), "plan_sha256": expected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/experiment/fair_reward_training_v1.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/seed_checkpoint_manifest.json")
    args = parser.parse_args()
    os.nice(max(0, 19 - os.nice(0)))
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    load_api(args.source_root.resolve())
    if args.mode == "prepare":
        prepare(args.source_root.resolve(), args.output_root.resolve(), args.config.resolve(), args.manifest.resolve())
    else:
        execute(args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
