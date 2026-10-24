"""Read-only audit of the decision opportunities behind saved reward curves.

Reads a completed long fair-reward run, never trains or selects checkpoints.
The only runtime probe constructs the frozen default environment without reset
or step, to check its cache-capacity setting. Output is create-only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen_environment_probe(source):
    # A separate interpreter prevents imports from the current checkout from
    # replacing the frozen scientific implementation. No reset/step is called.
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.data.model_catalog.adapter_catalog import AdapterCatalog
catalog = AdapterCatalog.from_json(Path(sys.argv[1]) / 'src/data/model_catalog/sample_model_catalog.json')
env = VecWorkflowCoreEnv(adapter_catalog=catalog)
print(json.dumps({'capacity_enabled': env._cache_capacity_enabled(),
                  'profile': env._cache_capacity_profile,
                  'reset_calls': 0, 'step_calls': 0}))
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return json.loads(subprocess.check_output(
        [sys.executable, "-B", "-c", code, str(source)], cwd=source,
        env=environment, text=True, timeout=60,
    ))


def reduce_service_chain(root, inputs):
    """Keep the small D/E service audit separate from the long-run population."""
    receipt_path = root / "final_validation.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "PASS", "Service-chain audit did not pass")
    inputs[str(receipt_path)] = sha(receipt_path)
    for filename in ["physical_service_and_identity_register.csv", "all_service_step_register.csv"]:
        path = root / filename
        require(sha(path) == receipt["local_files_sha256"][filename], "Service ledger changed")
        inputs[str(path)] = sha(path)
    physical = read_csv(root / "physical_service_and_identity_register.csv")
    service = {r["id"]: r for r in read_csv(root / "all_service_step_register.csv")}
    require(len(physical) == len(service) == 4704, "Unexpected saved service-audit population")
    require(len({r["id"] for r in physical}) == len(physical), "Duplicate service identity")
    groups = defaultdict(Counter)
    for row in physical:
        counterpart = service[row["id"]]
        require(float(row["native_reward"]) == float(counterpart["reward"])
                and row["actual_action"] == counterpart["actual_action"], "Service ledgers disagree")
        components = json.loads(row["reward_components"])
        total = sum(components[k] for k in ["positive_offset", "service_reward", "continuity_bonus", "mechanism_exploration_bonus"])
        total -= sum(components[k] for k in ["delay_penalty", "cache_miss_penalty", "migration_cost", "constraint_penalty"])
        require(abs(total - float(row["native_reward"])) <= 1e-8, "Service reward accounting differs")
        group = groups[row["cell"], row["phase"]]
        group["rows"] += 1
        group["transfer_events"] += int(row["transfer"])
        group["prepare_requests"] += row["prepare"] == "True"
        group["realized_prepare_events"] += row["realized"] == "True"
        group["prior_step_prepare_realizations"] += row["realized"] == "True" and row["age"] != "" and int(row["age"]) > 0
        group["subject_changes"] += row["subject_changed"] == "True"
        group["failure_" + row["failure_class"]] += 1
    return {"artifact_run_id": root.name,
            "scope": "4704 saved D/E preflight transitions only; not the full long training experiment",
            "all_reward_and_cross_ledger_checks": len(physical),
            "groups": [{"cell": cell, "phase": phase, **counts} for (cell, phase), counts in sorted(groups.items())]}


def audit(root, output, project, service_root=None):
    require(not output.exists(), "Output must not exist")
    plan = json.loads((root / "frozen_plan.json").read_text())
    projected = {k: v for k, v in plan.items() if k != "plan_sha256"}
    payload = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    require(hashlib.sha256(payload.encode()).hexdigest() == plan["plan_sha256"], "Plan changed")
    source = Path(plan["source_root"])
    source_git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    require(source_git == plan["source_commit"], "Frozen source HEAD changed")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"],
                   cwd=source, check=True, stdout=subprocess.DEVNULL)
    protected = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", *plan["protected_files"]], cwd=project)
    require(hashlib.sha256(protected).hexdigest() == plan["protected_diff_sha256"], "User diff changed")
    inputs = {}
    for filename, key in [("run_fair_reward_training.py", "base_runner_sha256"),
                          ("run_long_fair_reward_training.py", "runner_sha256"),
                          ("diagnose_sagh_flat_reward.py", "diagnostic_observer_sha256")]:
        path = source / "scripts" / filename
        require(sha(path) == plan[key], f"Frozen script changed: {filename}")
        inputs[str(path)] = sha(path)
    prior_path = root / "completion_audit.json"
    prior = json.loads(prior_path.read_text())
    require(prior["audit_status"] == "PASS" and prior["plan_sha256"] == plan["plan_sha256"],
            "Missing matching prior completion audit")
    inputs[str(prior_path)] = sha(prior_path)

    tree = ast.parse((source / "scripts/run_fair_reward_training.py").read_text())
    episode_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_episode")
    constructors = [n for n in ast.walk(episode_function) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "VecWorkflowCoreEnv"]
    require(len(constructors) == 1, "Unrecognized environment constructor")
    constructor = constructors[0]
    require(all(k.arg is not None for k in constructor.keywords), "Cannot audit dynamic constructor kwargs")
    require(all(k.arg != "cache_capacity_profile" for k in constructor.keywords), "Capacity profile supplied; review this protocol")
    episode_calls = Counter(n.func.id for n in ast.walk(episode_function)
                            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    require(all(episode_calls[name] >= 1 for name in ["clone_frames", "clone_rsu_state", "clone_workflow_state"]),
            "Unrecognized episode reset protocol")
    default = frozen_environment_probe(source)
    require(default["capacity_enabled"] is False, "Default environment has a capacity constraint")
    core_path = source / "src/envs/core/vec_workflow_core_env.py"
    inputs[str(core_path)] = sha(core_path)

    points = list(range(0, plan["training_steps"] + 1,
                        plan["steps_per_update"] * plan["evaluate_every_updates"]))
    expected_validation = {(w["window_id"], wf) for w in plan["windows"]
                           if w["split"] == "validation" for wf in plan["workflow_ids"]}
    expected_train = {(w["window_id"], wf) for w in plan["windows"]
                      if w["split"] == "train" for wf in plan["workflow_ids"]}
    finals, native, checkpoint_rows, cycle_rows, episode_counts = {}, {}, [], [], []
    validation_rows = trace_steps = trace_files = 0
    for name in plan["agents"]:
        for seed in plan["seeds"]:
            cell = root / "cells" / name / f"seed{seed}"
            done = json.loads((cell / "complete.json").read_text())
            require(done["status"] == "COMPLETED" and done["plan_sha256"] == plan["plan_sha256"], "Incomplete cell")
            require(done["training_steps"] == plan["training_steps"], "Unequal training budget")
            for filename, key in [("training.csv", "training_csv_sha256"), ("evaluation.csv", "evaluation_csv_sha256")]:
                path = cell / filename
                require(sha(path) == done[key], f"Saved CSV changed: {path}")
                inputs[str(path)] = sha(path)
            training = read_csv(cell / "training.csv")
            require(len(training) == done["episodes"], "Episode count mismatch")
            episode_counts.append(len(training))
            groups = defaultdict(list)
            for row in training:
                groups[int(row["cycle"])].append(row)
            require(set(groups) == set(range(1, plan["training_cycles"] + 1)), "Missing cycle")
            for cycle, rows in groups.items():
                require({(r["window_id"], r["workflow_id"]) for r in rows} == expected_train, "Train support differs")
                exposures = Counter()
                for row in rows:
                    exposures[row["window_id"], row["workflow_id"]] += int(row["episode_steps"])
                require(set(exposures.values()) == {plan["steps_per_training_unit"]}, "Unequal unit exposure")
                total = sum(float(r["total_reward"]) for r in rows)
                cycle_rows.append({"agent": name, "seed": seed, "cycle": cycle,
                                   "episodes": len(rows), "distinct_window_workflow_pairs": len(exposures),
                                   "steps": sum(exposures.values()), "mean_episode_reward": total / len(rows),
                                   "reward_per_interaction": total / sum(exposures.values())})
            evaluations = read_csv(cell / "evaluation.csv")
            validation_rows += len(evaluations)
            by_point = defaultdict(dict)
            for row in evaluations:
                unit = row["window_id"], row["workflow_id"]
                point = int(row["training_steps"])
                require(unit not in by_point[point], "Duplicate evaluation unit")
                require(row["agent"] == name and int(row["seed"]) == seed, "Evaluation identity differs")
                by_point[point][unit] = row
            require(set(by_point) == set(points), "Missing checkpoint")
            native[name, seed] = by_point
            for point in points:
                require(set(by_point[point]) == expected_validation, "Validation support differs")
                path = cell / f"validation_actions_{point:07d}.jsonl"
                inputs[str(path)] = sha(path)
                with path.open() as stream:
                    trace = [json.loads(line) for line in stream]
                trace_files += 1
                trace_steps += len(trace)
                indexed, totals = {}, defaultdict(float)
                for row in trace:
                    key = row["window_id"], row["workflow_id"], int(row["step"])
                    require(key not in indexed, "Duplicate recorded trajectory coordinate")
                    indexed[key] = int(row["action"]), float(row["reward"])
                    totals[key[:2]] += float(row["reward"])
                require(set(totals) == expected_validation, "Trace support differs")
                require(all(abs(totals[unit] - float(row["total_reward"])) <= 1e-6
                            for unit, row in by_point[point].items()), "Trace reward accounting mismatch")
                checkpoint_rows.append({"agent": name, "seed": seed, "training_steps": point,
                                        "reward_mean": statistics.mean(float(r["total_reward"]) for r in by_point[point].values()),
                                        "completed_fraction": statistics.mean(r["completed"] == "True" for r in by_point[point].values()),
                                        "action_counts": json.dumps(dict(Counter(r[0] for r in indexed.values())), sort_keys=True)})
                if point == plan["training_steps"]:
                    finals[name, seed] = indexed

    algorithm_rows = []
    for name in plan["agents"]:
        before = [r["reward_mean"] for r in checkpoint_rows if r["agent"] == name and r["training_steps"] == 0]
        after = [r["reward_mean"] for r in checkpoint_rows if r["agent"] == name and r["training_steps"] == plan["training_steps"]]
        final_units = [r for seed in plan["seeds"] for r in native[name, seed][plan["training_steps"]].values()]
        exact = sum(finals[name, seed] == finals["ppo", seed] for seed in plan["seeds"])
        paired_deltas = [float(native["sa_ghmappo", seed][plan["training_steps"]][unit]["total_reward"])
                         - float(row["total_reward"])
                         for seed in plan["seeds"] for unit, row in native[name, seed][plan["training_steps"]].items()]
        algorithm_rows.append({"agent": name, "initial_reward_mean": statistics.mean(before),
                               "final_reward_mean": statistics.mean(after),
                               "reward_change": statistics.mean(after) - statistics.mean(before),
                               "final_reward_seed_sd": statistics.stdev(after),
                               "final_exact_action_and_reward_matches_to_ppo": exact,
                               "sa_higher_final_paired_units": sum(d > 1e-8 for d in paired_deltas),
                               "sa_tied_final_paired_units": sum(abs(d) <= 1e-8 for d in paired_deltas),
                               "sa_lower_final_paired_units": sum(d < -1e-8 for d in paired_deltas),
                               "final_completion": statistics.mean(r["completed"] == "True" for r in final_units),
                               "final_continuity": statistics.mean(float(r["workflow_continuity_rate"]) for r in final_units),
                               "final_backhaul": statistics.mean(float(r["backhaul_traffic_cost"]) for r in final_units)})
    service_summary = reduce_service_chain(service_root, inputs) if service_root else None
    output.mkdir(parents=True)
    write_csv(output / "algorithm_results.csv", algorithm_rows)
    write_csv(output / "checkpoint_results.csv", checkpoint_rows)
    write_csv(output / "training_cycle_results.csv", cycle_rows)
    report = {
        "reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "literature_cutoff": "2026-09-28 (not reassessed; no novelty review)",
        "target_venue": "CSCWD2027 development", "artifact_run_id": root.name,
        "policy_version": "tmc_review_policy_v3_20260621", "git_commit": source_git,
        "workspace_git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "evidence_level": "READ_ONLY_NATIVE_CSV_TRACE_REDUCTION_FROZEN_SOURCE_AND_DEFAULT_CONSTRUCTOR_PROBE",
        "verdict": "EXISTING_CURVES_DO_NOT_ESTABLISH_MAIN_METHOD_SUPERIORITY",
        "new_training_gain": "UNVERIFIED", "formal_holdout": "UNVERIFIED", "novelty_score": "N/S",
        "checks": {"status": "PASS", "cells": len(episode_counts), "validation_rows": validation_rows,
                   "trace_files": trace_files, "trace_steps": trace_steps, "training_cycles": len(cycle_rows),
                   "new_environment_resets": 0, "new_environment_steps": 0, "new_optimizer_steps": 0},
        "environment": {"runner_passes_capacity_profile": False, "default_constructor_probe": default,
                        "fresh_frame_rsu_workflow_clone_per_episode": True,
                        "episode_step_cap": plan["max_episode_steps"], "window_frames": plan["window_length"],
                        "train_window_workflow_pairs": len(expected_train),
                        "training_cycles_per_cell": plan["training_cycles"],
                        "episode_count_min": min(episode_counts), "episode_count_max": max(episode_counts)},
        "algorithm_results": algorithm_rows,
        "small_preflight_service_chain": service_summary,
        "limits": ["Trace equality does not establish equal networks or action probabilities.",
                   "Different action sequences may produce equal service outcomes; no causal rule-removal ablation was run.",
                   "Capacity-disabled finding is scoped to this independent reward experiment, not formal capacity benchmarks.",
                   "Per-episode resets are a standard episodic design; their limitation here is the untested persistent cache tradeoff.",
                   "Prior completion audit referenced; original data files and checkpoint weights are not rehashed in this pass.",
                   "Repeated exposure is training, not independent evaluation sample size.",
                   "Pooled paired-unit win/tie/loss counts are descriptive repeats across seeds, not independent statistical clusters.",
                   "New workload design can test differentiation but cannot guarantee distinct or favorably ranked curves."],
        "audit_script_sha256": sha(Path(__file__)),
        "command_argv": [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        "source_files_sha256": inputs,
        "next_experiment_prerequisites": [
            "Validate strict subject/prediction binding on the real provider and every failure consumer, retaining all exposure outcomes.",
            "Predeclare a common workload from mobility/workflow/resource features, without reading outcome rewards to choose cases.",
            "Expose finite cache competition and prepare-before-transfer-before-later-service chains; retain negative and censored cases.",
            "Check execution-distribution controllability and supervised target reachability on saved states before additional long training.",
            "Use equal interaction budgets, all fixed seeds/checkpoints, dense early validation and service metrics beside native reward.",
            "Keep short pilot and protocol changes separate from the original curves; freeze the long-run protocol before execution."],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "checks": report["checks"],
                      "environment": report["environment"], "algorithms": algorithm_rows}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--service-chain-root", type=Path,
                        help="Optional separate, previously audited D/E service-chain evidence")
    args = parser.parse_args()
    audit(args.run_root.resolve(), args.output_root.resolve(), Path(__file__).resolve().parents[1],
          args.service_chain_root.resolve() if args.service_chain_root else None)


if __name__ == "__main__":
    main()
