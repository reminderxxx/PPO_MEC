"""Read-only checkpoint replay and paired-state SAGH action diagnosis.

Hooks observe existing outputs and return the original objects unchanged. No
learning, alternate policy configuration, or checkpoint writes are performed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import run_fair_reward_training as fair


STAGES = ["_apply_digital_twin_policy_prior", "_apply_opportunity_constrained_policy",
          "_apply_backhaul_aware_policy", "_apply_net_advantage_prepare_gate",
          "_apply_sparse_handoff_recovery_prior", "_apply_continuity_guard",
          "_apply_service_completion_gate", "_apply_event_logit_sharpening"]


def snapshot(agent, output, mask):
    scores = agent._hierarchical_env_action_scores(output)
    masked = agent._masked_flat_logits(scores, mask)
    return {"logits": {head: output[head + "_logits"].detach().cpu().tolist()
                       for head in ("slow", "fast", "event")},
            "env_probs": fair.torch.softmax(masked, dim=-1).detach().cpu().tolist(),
            "env_argmax": int(masked.argmax().item())}


def observe(agent):
    """Install transparent observers, including the unadjusted network output."""
    trace = {"active": False, "stages": []}
    forward = agent._network.forward_single

    def forward_observed(*args, **kwargs):
        result = forward(*args, **kwargs)
        if trace["active"]:
            trace["stages"].append({"stage": "network", **snapshot(agent, result, trace["mask"])})
        return result

    agent._network.forward_single = forward_observed
    for method in STAGES:
        original = getattr(agent, method)

        def observed(*args, _original=original, _name=method, **kwargs):
            result = _original(*args, **kwargs)
            if trace["active"]:
                trace["stages"].append({"stage": _name, **snapshot(agent, result, trace["mask"])})
            return result

        setattr(agent, method, observed)
    act = agent.act

    def act_observed(observation, info=None):
        trace.update(active=True, mask=agent._extract_action_mask(info), stages=[])
        try:
            action, diagnostics = act(observation, info)
            diagnostics = dict(diagnostics)
            diagnostics["checkpoint_probe"] = copy.deepcopy(trace["stages"])
            return action, diagnostics
        finally:
            trace["active"] = False

    agent.act = act_observed
    return agent


def restored(p, seed, checkpoint):
    payload = fair.torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = fair.fresh_agent("sa_ghmappo", p["agent_configs"]["sa_ghmappo"][str(seed)], seed)
    agent._network.load_state_dict(payload["network_state_dict"])
    agent._update_count = payload["update_count"]
    agent._deterministic_action = True
    agent._network.eval()
    return observe(agent)


def action_record(row):
    info = row["action_info"]
    return {"action": int(row["action"]), "reward": float(row["reward"]),
            "head_actions": info["head_actions"], "raw_env_action": info["raw_env_action"],
            "projected_env_action": info["projected_env_action"],
            "env_probs": info["env_action_probs"], "env_probs_scope": info.get("env_action_probability_scope", "legacy_pre_mapping"),
            "probability_contract": info.get("probability_contract"),
            "sampled_base_action": info.get("sampled_base_action"), "sampled_path_log_prob": info["log_prob"],
            "mask": info["action_mask"],
            "stages": info["checkpoint_probe"], "guard_action_delta": info["guard_action_delta"],
            "option_gate": info["option_gate"], "aggregation_reason": info["aggregation_reason"],
            "continuity_guard": info["continuity_guard"],
            "cache_warm_start_guard": info["cache_warm_start_guard"],
            "idle_popularity_fallback": info["idle_popularity_fallback"],
            "deterministic_temporal_smoothing": info["deterministic_temporal_smoothing"]}


def compare_records(a, b):
    return {"actual_action_changed": a["action"] != b["action"],
            "head_actions_changed": a["head_actions"] != b["head_actions"],
            "network_argmax_changed": a["stages"][0]["env_argmax"] != b["stages"][0]["env_argmax"],
            "network_probability_l1": sum(abs(x-y) for x, y in zip(a["stages"][0]["env_probs"], b["stages"][0]["env_probs"])),
            "proposal_probability_l1": sum(abs(x-y) for x, y in zip(a["env_probs"], b["env_probs"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_root.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError("Diagnostic is create-only")
    os.nice(max(0, 19-os.nice(0)))
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    p = json.loads((root / "frozen_plan.json").read_text())
    source = Path(p["source_root"])
    expected = p.pop("plan_sha256")
    if fair.semantic_hash(p) != expected:
        raise ValueError("Source plan changed")
    p["plan_sha256"] = expected
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], cwd=source, check=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != p["source_commit"]:
        raise ValueError("Source commit changed")
    fair.load_api(source)
    bundles = [fair.window_bundle(source, p, w["frame_offset"]) for w in p["windows"] if w["split"] == "validation"]
    workflows = fair.build_workflows(workflow_csv_path=p["workflow_path"], max_workflows=p["max_workflows"],
                                    workflow_selector="ordered", min_tasks=5, max_tasks=20, random_seed=7)
    out.mkdir(parents=True)
    fair.write_json(out / "protocol.json", {"parent_plan_sha256": expected, "source_commit": p["source_commit"],
                    "source_root": str(source), "script_sha256": fair.sha(__file__),
                    "checkpoint_updates": [0, 12, 24], "seeds": p["seeds"],
                    "validation_units_per_checkpoint": len(bundles)*len(workflows),
                    "learning_calls": 0, "policy_changes": "none; transparent output observers only", "started_at": time.time()})
    episodes, paired, checkpoints, stage_counts = [], [], [], []
    for seed in p["seeds"]:
        references = []
        source_cell = root / "cells/sa_ghmappo" / f"seed{seed}"
        for update in (0, 12, 24):
            checkpoint = source_cell / ("initial.pt" if update == 0 else f"update_{update:04d}.pt")
            agent = restored(p, seed, checkpoint)
            checkpoints.append({"seed": seed, "update": update, "path": str(checkpoint), "sha256": fair.sha(checkpoint)})
            counter = Counter()
            traces = []
            for i, bundle in enumerate(bundles):
                fair.yield_resources(out)
                for j, workflow in enumerate(workflows):
                    metadata = {"phase": "validation", "training_steps": update*256,
                                "window_id": bundle.rsu_metadata["window_id"], "workflow_id": workflow.workflow_id}
                    summary, rollout = fair.make_episode(source, p, agent, bundle, workflow, seed+10000+i*2+j,
                                                          p["max_episode_steps"], metadata, False)
                    records = [action_record(row) for row in rollout]
                    episodes.append({"seed": seed, "update": update, "unit": i*2+j,
                                     "total_reward": summary["reward_breakdown"]["total"]["sum"],
                                     "completed": bool(summary["episode_status"].get("completed", False)),
                                     "steps": len(rollout), "actions": json.dumps([r["action"] for r in records])})
                    for k, (row, record) in enumerate(zip(rollout, records)):
                        traces.append({"seed": seed, "update": update, "unit": i*2+j, "step": k, **record})
                        counter["states"] += 1
                        counter[f"executed_action_{record['action']}"] += 1
                        counter["network_to_final_argmax_changed"] += record["stages"][0]["env_argmax"] != record["action"]
                        previous = record["stages"][0]["env_argmax"]
                        for stage in record["stages"][1:]:
                            counter[stage["stage"] + "_argmax_change"] += previous != stage["env_argmax"]
                            previous = stage["env_argmax"]
                        counter["post_logits_to_actual_change"] += previous != record["action"]
                        if update == 0:
                            references.append((copy.deepcopy(row["observation"]), copy.deepcopy(row["decision_info"]), record, i*2+j, k))
            with (out / f"trajectory_seed{seed}_update{update:04d}.jsonl").open("w") as f:
                for row in traces:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stage_counts.append({"seed": seed, "update": update, **dict(counter)})
            # Pair every model on the entire initial checkpoint state/memory set.
            probe = restored(p, seed, checkpoint)
            for observation, info, initial, unit, step in references:
                info = copy.deepcopy(info)
                info["run_metadata"]["training_steps"] = update*256
                action, action_info = probe.act(observation, info)
                record = action_record({"action": action, "reward": 0., "action_info": action_info})
                paired.append({"seed": seed, "update": update, "unit": unit, "step": step,
                               **compare_records(initial, record)})
            print(f"diagnosed seed={seed} update={update} states={counter['states']}", flush=True)
    fair.csv_write(out / "episode_replay.csv", episodes)
    fair.csv_write(out / "paired_states.csv", paired)
    # Stage dictionaries share the same keys, including zero counts.
    fair.csv_write(out / "stage_counts.csv", stage_counts)
    summary = []
    for seed in p["seeds"]:
        for update in (0, 12, 24):
            rows = [r for r in paired if r["seed"] == seed and r["update"] == update]
            eps = [r for r in episodes if r["seed"] == seed and r["update"] == update]
            summary.append({"seed": seed, "update": update, "paired_states": len(rows),
                            "actual_action_changes": sum(r["actual_action_changed"] for r in rows),
                            "head_action_changes": sum(r["head_actions_changed"] for r in rows),
                            "network_argmax_changes": sum(r["network_argmax_changed"] for r in rows),
                            "network_probability_l1_mean": sum(r["network_probability_l1"] for r in rows)/len(rows),
                            "proposal_probability_l1_mean": sum(r["proposal_probability_l1"] for r in rows)/len(rows),
                            "evaluation_reward_mean": sum(r["total_reward"] for r in eps)/len(eps),
                            "completion_rate": sum(r["completed"] for r in eps)/len(eps)})
    fair.csv_write(out / "summary.csv", summary)
    fair.write_json(out / "complete.json", {"status": "COMPLETED", "finished_at": time.time(),
                    "checkpoints": checkpoints, "summary": summary,
                    "files": {f.name: fair.sha(f) for f in out.iterdir() if f.is_file()}})


if __name__ == "__main__":
    main()
