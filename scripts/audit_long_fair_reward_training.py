"""Offline completion audit; no policy rollout, training or checkpoint selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import run_fair_reward_training as fair
from plot_long_fair_reward_training import validate_cell


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = root / "completion_audit.json"
    if output.exists():
        raise FileExistsError("Audit is create-only; inspect the existing audit first")
    os.nice(max(0, 19-os.nice(0)))
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    p = json.loads((root / "frozen_plan.json").read_text())
    copy = dict(p)
    copy.pop("plan_sha256")
    check(fair.semantic_hash(copy) == p["plan_sha256"], "Plan changed")
    for filename, key in [("run_fair_reward_training.py", "base_runner_sha256"),
                          ("run_long_fair_reward_training.py", "runner_sha256"),
                          ("diagnose_sagh_flat_reward.py", "diagnostic_observer_sha256")]:
        check(fair.sha(Path(__file__).with_name(filename)) == p[key], f"Source script changed: {filename}")
    check(fair.sha(p["manifest_path"]) == p["manifest_sha256"], "Source config manifest changed")
    source = Path(p["source_root"])
    check(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == p["source_commit"], "Source commit changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], cwd=source, check=True)
    protected = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *p["protected_files"]], cwd=fair.PROJECT)
    digest = hashlib.sha256(protected).hexdigest()
    check(digest == p["protected_diff_sha256"], "Protected mainline diff changed; preserve and report, do not reset it")
    for key, identity in p["data_files"].items():
        check(Path(p[key]).stat().st_size == identity["size"] and fair.sha(p[key]) == identity["sha256"], f"Data changed: {key}")
    for item in p["configuration_sources"]:
        check(fair.sha(item["configuration_source_path"]) == item["source_checkpoint_sha256"], "Old configuration checkpoint changed")
    fair.load_api(source)
    fair.intervals_disjoint(p["windows"])
    for w in p["windows"]:
        b = fair.window_bundle(source, p, w["frame_offset"])
        identity = fair.window_identity(b)
        check(identity == {k: w[k] for k in identity}, "Actual window identity differs")
        content = [{**{k: v for k, v in f.items() if k != "vehicles"}, "vehicles": [v.to_dict() for v in f["vehicles"]]} for f in b.frames]
        check(fair.semantic_hash(content) == w["frame_content_sha256"], "Actual window frames differ")
    workflows = fair.build_workflows(workflow_csv_path=p["workflow_path"], max_workflows=p["max_workflows"],
                                    workflow_selector="ordered", min_tasks=5, max_tasks=20, random_seed=7)
    check(fair.semantic_hash([w.to_dict() for w in workflows]) == p["workflow_content_sha256"], "Actual workflows differ")
    cells, inventory = [], []
    total_updates = p["training_steps"]//p["steps_per_update"]
    checkpoint_updates = [0, *range(p["evaluate_every_updates"], total_updates+1, p["evaluate_every_updates"])]
    for name in p["agents"]:
        for seed in p["seeds"]:
            cell = root / "cells" / name / f"seed{seed}"
            rows, evaluations, means, completion = validate_cell(cell, p, name, seed)
            identity = json.loads((cell / "identity.json").read_text())
            check(identity["agent"] == name and identity["seed"] == seed and identity["plan_sha256"] == p["plan_sha256"], "Cell identity changed")
            check(identity["fresh_initialization"] and identity["replay_initial_size"] == 0, "Initialization is not fresh")
            initial, final = None, None
            for update in checkpoint_updates:
                path = cell / ("initial.pt" if update == 0 else f"update_{update:04d}.pt")
                ck = fair.torch.load(path, map_location="cpu", weights_only=False)
                check(ck["agent_name"] == name and ck["update_count"] == update, "Checkpoint update/agent mismatch")
                state = ck["network_state_dict"]
                if update == 0:
                    check(not ck["optimizer_state_dict"]["state"], "Initial optimizer is not empty")
                    initial = state
                check(set(state) == set(initial), "Checkpoint architecture changed")
                for key, value in state.items():
                    check(value.shape == initial[key].shape and fair.torch.isfinite(value).all().item(), "Invalid checkpoint tensor")
                final = state
                inventory.append({"cell": str(cell.relative_to(root)), "update": update, "sha256": fair.sha(path)})
            changed = [key for key in initial if not fair.torch.equal(initial[key], final[key])]
            l2 = sum(float((initial[k]-final[k]).float().pow(2).sum()) for k in changed)**.5
            learning = [json.loads(line)["learn_info"] for line in (cell / "updates.jsonl").open()]
            stability = {"accepted_optimizer_steps": sum(v.get("accepted_optimizer_steps", 0) for v in learning),
                         "rejected_optimizer_steps": sum(v.get("rejected_optimizer_steps", 0) for v in learning),
                         "optimizer_backtracks": sum(v.get("optimizer_backtracks", 0) for v in learning),
                         "initial_log_prob_error_max": max(v.get("initial_log_prob_error_max", 0.) for v in learning),
                         "accepted_policy_kl_max": max(v.get("accepted_policy_kl_max", 0.) for v in learning)}
            cells.append({"agent": name, "seed": seed, "steps": p["training_steps"], "updates": total_updates,
                          "cycles": p["training_cycles"], "episodes": len(rows), "validation_rows": len(evaluations),
                          "initial_validation_reward": means[0], "final_validation_reward": means[-1],
                          "initial_completion_rate": completion[0], "final_completion_rate": completion[-1],
                          "changed_network_tensors": len(changed), "network_l2_change": l2,
                          "stability_diagnostics": stability})
    report = {"audit_status": "PASS", "reviewed_at": datetime.now().astimezone().isoformat(),
              "literature_cutoff": "2026-09-28 (not reassessed)", "target_venue": "CSCWD 2027 development",
              "artifact_run_id": root.name, "policy_version": "tmc_review_policy_v3_20260621", "git_commit": p["source_commit"],
              "evidence_level": "MATCHED_DEVELOPMENT_CHECKPOINTS_NATIVE_RECEIPTS_AND_ACTION_TRACES_AUDITED",
              "formal_holdout": "UNVERIFIED", "plan_sha256": p["plan_sha256"], "protected_diff_sha256": digest,
              "cells": cells, "checkpoint_inventory": inventory,
              "limitations": ["Previously explored development windows, no untouched holdout or novelty review",
                              "Native architectures, gradient epochs, KL early stopping and compute costs differ",
                              "Reward plateau alone does not prove training gains or policy convergence"]}
    fair.write_json(output, report)
    print(json.dumps({"audit_status": "PASS", "cells": len(cells), "checkpoints": len(inventory)}))


if __name__ == "__main__":
    main()
