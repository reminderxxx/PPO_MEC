"""Read-only event cost decomposition for the frozen long-budget run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_sa_behavior import _bool, _distance, _public_state_hash
from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import _load_inputs, _load_json
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv, _network_seconds

B_ROOT = Path("/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC")
SOURCE = B_ROOT / "artifacts/experiments/cscwd_causal_strong_baselines_budget_extension_20261009_v1"
TERMINAL = B_ROOT / "artifacts/experiments/cscwd_causal_strong_baselines_budget_extension_20261009_v1_supervisor/terminal_receipt.json"
OUTPUT = ROOT / "artifacts/analysis/cscwd_sa_long_budget_cost_diagnosis_20261009_v1"
PLAN = ROOT / "docs/project/cscwd_sa_long_budget_cost_diagnosis_plan_20261009.md"
BASE_DESIGN = ROOT / "configs/experiment/calibrated_workflow_strong_baselines_development_v2_prefix_only.json"
SCIENTIFIC_COMMIT = "d25ebcded6b43b69b82bae825b035adc1d6f19c4"
CAUSAL_COMMIT = "a08869388f9962149198054b5f8a0d6dd4d07eae"
METHODS = ("sa_ghmappo", "mappo", "ppo", "dt_handoff_drl")
COMPONENTS = (
    "node_compute", "vehicle_fallback", "failed_service_wait", "model_load",
    "model_network", "state_restore", "state_network", "input_network", "dag_recompute",
)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _close(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-6)


def _verify_inputs() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    manifest = _load_json(SOURCE / "run_manifest.json")
    receipt = _load_json(SOURCE / "completion_receipt.json")
    terminal = _load_json(TERMINAL)
    if manifest["git_commit"] != SCIENTIFIC_COMMIT or manifest["causal_implementation_commit"] != CAUSAL_COMMIT:
        raise RuntimeError("long-budget scientific identity mismatch")
    if receipt.get("status") != "complete" or receipt.get("scientific_execution_complete") is not True or receipt.get("learned_environment_steps") != 115200:
        raise RuntimeError("scientific training did not complete")
    if terminal.get("child_receipt", {}).get("scientific_execution_complete") is not True:
        raise RuntimeError("supervisor lacks scientific child completion receipt")
    if terminal.get("status") != "FAIL" or terminal.get("return_code") != 1:
        raise RuntimeError("unexpected supervisor terminal; inspect before proceeding")
    failure = _load_json(SOURCE / "analysis_failure_receipt.json")
    if failure.get("error_type") != "TypeError" or "zip() takes no keyword arguments" not in failure.get("error", ""):
        raise RuntimeError("postprocess failure differs from documented Python compatibility issue")
    science_files = (
        "run_manifest.json", "completion_receipt.json", "evaluation_rows.csv", "behavior_ledger.csv",
        "training_summary.json", "checkpoint_selection.json",
        "training_signal_rows.csv", "optimizer_step_records.csv",
    )
    inventory = {r["path"]: r for r in _load_json(SOURCE / "artifact_integrity.json")["files"]}
    for name in science_files:
        path, entry = SOURCE / name, inventory[name]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]) or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"scientific input hash mismatch: {name}")
    training = _load_json(SOURCE / "training_summary.json")
    if len(training) != 20:
        raise RuntimeError("expected 20 selected learned cells")
    for cell in training:
        name = cell["selected_checkpoint"]
        if _sha256(SOURCE / name) != cell["selected_checkpoint_sha256"] or _sha256(SOURCE / name) != inventory[name]["sha256"]:
            raise RuntimeError(f"selected checkpoint hash mismatch: {name}")
    science_modules = [
        "src/envs/core/calibrated_continuous_workflow_env.py",
        "src/envs/core/causal_rsu_predictor.py", "src/encoders",
        "src/agents", "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json",
        "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json",
    ]
    diff = subprocess.run(["git", "diff", "--quiet", CAUSAL_COMMIT, SCIENTIFIC_COMMIT, "--", *science_modules], cwd=ROOT, check=False)
    if diff.returncode:
        raise RuntimeError("environment/agent/encoder scientific implementation differs from causal baseline")
    if manifest["base_config"]["sha256"] != _sha256(ROOT / manifest["base_config"]["path"]):
        raise RuntimeError("base config identity differs")
    if manifest["workload_manifest"]["sha256"] != _sha256(ROOT / manifest["workload_manifest"]["path"]):
        raise RuntimeError("workload source identity differs")
    config, splits, identity = _load_inputs(_load_json(BASE_DESIGN))
    if identity["public_prefix_suffix_tamper_comparisons"] != 474:
        raise RuntimeError("causal public input boundary drift")
    evaluation, ledger = _csv(SOURCE / "evaluation_rows.csv"), _csv(SOURCE / "behavior_ledger.csv")
    curve = _csv(SOURCE / "checkpoint_selection_curve.csv")
    if len(evaluation) != 400 or len(ledger) != 2811 or len(curve) != 80:
        raise RuntimeError("evaluation/behavior/selection count drift")
    candidates = _load_json(SOURCE / "checkpoint_selection.json")
    candidate_keys = {(str(r["method"]), str(r["seed"]), int(r["update_index"])) for r in candidates}
    curve_keys = {(r["method"], r["seed"], int(r["update_index"])) for r in curve}
    if len(candidates) != 80 or candidate_keys != curve_keys:
        raise RuntimeError("postprocessed selection curve does not match scientific candidates")
    selected_updates = {(str(r["method"]), str(r["seed"])): int(r["selected_update"]) for r in training}
    for row in curve:
        if (row["selected"] == "True") != (int(row["update_index"]) == selected_updates[(row["method"], row["seed"])]):
            raise RuntimeError("postprocessed selected checkpoint marker mismatch")
    if len({(r["split"], r["method"], r["seed"], r["design_id"]) for r in evaluation}) != 400:
        raise RuntimeError("duplicate evaluation identity")
    if {r["method"] for r in evaluation} != set(METHODS):
        raise RuntimeError("method set drift")
    if len(_csv(SOURCE / "training_signal_rows.csv")) != 115200 or len(_csv(SOURCE / "optimizer_step_records.csv")) != 15360:
        raise RuntimeError("training signal/optimizer count drift")
    return config, splits, evaluation, ledger, curve, manifest


def _replay(config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], evaluation: list[dict[str, str]], ledger: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances = {r["design_id"]: r for split in ("regression", "frozen_check") for r in splits[split]}
    episodes_ledger: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        episodes_ledger[(row["split"], row["method"], row["seed"], row["design_id"])].append(row)
    eval_map = {(r["split"], r["method"], r["seed"], r["design_id"]): r for r in evaluation}
    if set(episodes_ledger) != set(eval_map):
        raise RuntimeError("behavior/evaluation identity mismatch")
    events: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for key, records in sorted(episodes_ledger.items()):
        split, method, seed, design_id = key
        env = CalibratedContinuousWorkflowEnv(config, instances[design_id])
        observation, info = env.reset()
        victim_history: set[tuple[str, str]] = set()
        prepared: dict[str, int] = {}
        seen_prepare: Counter[tuple[str, str]] = Counter()
        component_totals = {component: 0.0 for component in COMPONENTS}
        first_crossing = None
        first_failed_step = None
        for row in sorted(records, key=lambda x: int(x["step_index"])):
            step = int(row["step_index"])
            if env.step_index != step or env.terminated:
                raise RuntimeError(f"episode step identity drift: {key} {step}")
            node = env._current_node()
            current, target = env._current_rsu_id(), env._predicted_handoff_target()
            current_ready = env._bundle_ready(current, str(node["required_adapter"]))
            target_ready = bool(target and env._bundle_ready(target, str(node["required_adapter"])))
            if current_ready != _bool(row["current_bundle_ready"]) or target_ready != _bool(row["target_bundle_ready"]):
                raise RuntimeError(f"cache readiness mismatch: {key} {step}")
            before_target = list(env.caches[target].residents) if target else []
            public_hash = _public_state_hash(observation, info)
            action = int(row["executed_action"])
            if not info["action_mask"][action]:
                raise RuntimeError("recorded action outside public mask")
            reuse_source_step = prepared.pop(current, None) if current_ready else None
            before_completed = len(env.completed)
            observation, _, terminated, truncated, info = env.step(action)
            event = info["transition"]
            if bool(event["service_completed"]) != _bool(row["service_completed"]) or bool(event["migration_success"]) != _bool(row["migration_success"]) or (len(env.completed) > before_completed) != _bool(row["progressed"]):
                raise RuntimeError(f"service/migration/progress mismatch: {key} {step}")
            for name in ("model_transfer_bytes", "state_transfer_bytes", "recompute_seconds", "clock_seconds_after"):
                if not _close(event[name], row[name]):
                    raise RuntimeError(f"event metric mismatch: {key} {step} {name}")
            actual_mbps = env._effective_link_mbps()
            fixed = float(config["link"]["fixed_seconds"])
            components = {
                "node_compute": float(node["compute_seconds"]) if event["service_completed"] else 0.0,
                "vehicle_fallback": float(config["vehicle"]["fallback_seconds"]) if action == 2 else 0.0,
                "failed_service_wait": float(config["objective"]["failed_service_seconds"]) if not event["service_completed"] else 0.0,
                "model_load": float(event["model_load_seconds"]),
                "model_network": _network_seconds(int(event["model_transfer_bytes"]), actual_mbps, fixed),
                "state_restore": float(event["state_restore_seconds"]),
                "state_network": _network_seconds(int(event["state_transfer_bytes"]), actual_mbps, fixed),
                "input_network": _network_seconds(int(event["input_transfer_bytes"]), actual_mbps, fixed),
                "dag_recompute": float(event["recompute_seconds"]),
            }
            if not _close(sum(components.values()), event["step_cost_seconds"]):
                raise RuntimeError(f"step cost not decomposed: {key} {step}")
            for name, value in components.items():
                component_totals[name] += value
            if event["deadline_miss_event"] and first_crossing is None:
                first_crossing = step
            if not event["service_completed"] and first_failed_step is None:
                first_failed_step = step
            cache_events = list(event["cache_events"])
            victims = [(str(item["rsu_id"]), str(victim)) for item in cache_events if item.get("committed") for victim in item.get("victims", [])]
            admissions = [(str(item["rsu_id"]), str(admitted)) for item in cache_events if item.get("committed") for admitted in item.get("admitted", [])]
            reloads = [pair for pair in admissions if pair in victim_history]
            victim_history.update(victims)
            repeated_prepare = False
            if action in (1, 4) and target:
                tag = (target, str(node["required_adapter"]))
                repeated_prepare = seen_prepare[tag] > 0 and int(event["model_transfer_bytes"]) > 0
                if any(item.get("committed") and str(item.get("rsu_id")) == target for item in cache_events):
                    seen_prepare[tag] += 1
                    prepared[target] = step
            after_target = list(env.caches[target].residents) if target else []
            recorded_forecast = json.loads(row["prediction_provenance"])
            distance = _distance(current, recorded_forecast["sequence"])
            future = env.instance["rsu_sequence"][step + 1 : step + 1 + int(config["prediction_horizon"])]
            actual_first = next((str(candidate) for candidate in future if str(candidate) != current), None)
            events.append({
                "split": split, "method": method, "seed": seed, "design_id": design_id, "step_index": step,
                "public_state_sha256": public_hash, "action": action,
                "raw_env_action": int(row["raw_env_action"]), "projected_env_action": int(row["projected_env_action"]),
                "projection_applied": _bool(row["projection_applied"]),
                "current_rsu": current, "target_rsu": target, "current_ready": current_ready, "target_ready": target_ready,
                "target_prepare_feasible": _bool(row["target_prepare_feasible"]),
                "past_handoff_now": bool(event["handoff"]), "predicted_distance": distance,
                "actual_first_handoff_target_posthoc": actual_first,
                "predicted_target_matches_actual_posthoc": target == actual_first if target and actual_first else None,
                "service_completed": bool(event["service_completed"]), "node_progressed": len(env.completed) > before_completed,
                "migration_prepare_committed": bool(event["migration_success"]),
                "target_residents_before": before_target, "target_residents_after": after_target,
                "target_resident_changed": before_target != after_target,
                "victims": victims, "admissions": admissions, "victim_reload_count": len(reloads),
                "prepared_target_reuse_source_step": reuse_source_step, "repeated_model_prepare": repeated_prepare,
                "deadline_crossing": bool(event["deadline_miss_event"]),
                "clock_seconds_after": float(event["clock_seconds_after"]),
                "step_cost_seconds": float(event["step_cost_seconds"]),
                "model_transfer_bytes": int(event["model_transfer_bytes"]),
                "state_transfer_bytes": int(event["state_transfer_bytes"]),
                "input_transfer_bytes": int(event["input_transfer_bytes"]),
                **components,
            })
            if terminated or truncated:
                if step != max(int(r["step_index"]) for r in records):
                    raise RuntimeError("premature replay termination")
                break
        original = eval_map[key]
        summary = env.summary()
        if not _close(summary["modeled_completion_seconds"], original["modeled_completion_seconds"]) or not _close(sum(component_totals.values()), original["modeled_completion_seconds"]):
            raise RuntimeError(f"episode cost mismatch: {key}")
        for outcome, source in (("workflow_completed", "workflow_completion_rate"), ("on_time_workflow_completed", "on_time_workflow_completion_rate"), ("service_failures", "service_failures")):
            if not _close(summary[outcome], original[source]):
                raise RuntimeError(f"episode outcome mismatch: {key} {outcome}")
        region = "on_time" if summary["on_time_workflow_completed"] else "late_completed" if summary["workflow_completed"] else "incomplete"
        episode_rows.append({
            "split": split, "method": method, "seed": seed, "design_id": design_id,
            "region": region, "completed": bool(summary["workflow_completed"]),
            "on_time": bool(summary["on_time_workflow_completed"]), "steps": len(records),
            "deadline_seconds": float(env.instance["deadline_seconds"]),
            "modeled_completion_seconds": float(summary["modeled_completion_seconds"]),
            "first_deadline_crossing_step": first_crossing, "first_failed_service_step": first_failed_step,
            "service_failures": int(summary["service_failures"]),
            "no_progress_steps": sum(not bool(e["node_progressed"]) for e in events[-len(records):]),
            "vehicle_fallback_actions": sum(e["action"] == 2 for e in events[-len(records):]),
            "action_3_current_missing": sum(e["action"] == 3 and not e["current_ready"] for e in events[-len(records):]),
            "action_4_current_missing": sum(e["action"] == 4 and not e["current_ready"] for e in events[-len(records):]),
            "target_resident_changes": sum(e["target_resident_changed"] for e in events[-len(records):]),
            "target_reuses": sum(e["prepared_target_reuse_source_step"] is not None for e in events[-len(records):]),
            "victim_reloads": sum(e["victim_reload_count"] for e in events[-len(records):]),
            "repeated_model_prepares": sum(e["repeated_model_prepare"] for e in events[-len(records):]),
            "model_transfer_bytes": int(summary["model_transfer_bytes"]),
            "state_transfer_bytes": int(summary["state_transfer_bytes"]),
            "input_transfer_bytes": int(summary["input_transfer_bytes"]),
            **component_totals,
        })
    if len(events) != 2811 or len(episode_rows) != 400:
        raise RuntimeError("replay count drift")
    return events, episode_rows


def _aggregate(events: list[dict[str, Any]], episodes: list[dict[str, Any]], curve: list[dict[str, str]]) -> dict[str, Any]:
    by_episode = {(r["split"], r["method"], r["seed"], r["design_id"]): r for r in episodes}
    event_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        event_groups[(e["split"], e["method"], e["seed"], e["design_id"])].append(e)
    first_divergences = []
    paired = []
    for sa in episodes:
        if sa["method"] != "sa_ghmappo":
            continue
        for peer_method in ("ppo", "dt_handoff_drl"):
            peer = by_episode[(sa["split"], peer_method, sa["seed"], sa["design_id"])]
            left = event_groups[(sa["split"], "sa_ghmappo", sa["seed"], sa["design_id"])]
            right = event_groups[(sa["split"], peer_method, sa["seed"], sa["design_id"])]
            first = next(((a, b) for a, b in zip(left, right) if a["action"] != b["action"]), None)
            first_divergences.append({
                "split": sa["split"], "seed": sa["seed"], "design_id": sa["design_id"], "comparator": peer_method,
                "sa_region": sa["region"], "first_divergence_step": first[0]["step_index"] if first else None,
                "same_public_state_at_first_divergence": first[0]["public_state_sha256"] == first[1]["public_state_sha256"] if first else None,
                "sa_first_action": first[0]["action"] if first else None,
                "peer_first_action": first[1]["action"] if first else None,
            })
            both = sa["completed"] and peer["completed"]
            row = {"split": sa["split"], "seed": sa["seed"], "design_id": sa["design_id"], "comparator": peer_method,
                   "sa_region": sa["region"], "peer_region": peer["region"], "both_completed": both,
                   "sa_completed": sa["completed"], "peer_completed": peer["completed"],
                   "sa_on_time": sa["on_time"], "peer_on_time": peer["on_time"]}
            for name in ("modeled_completion_seconds", *COMPONENTS, "model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes", "target_reuses", "victim_reloads", "no_progress_steps"):
                row[f"sa_{name}"] = sa[name]
                row[f"peer_{name}"] = peer[name]
                row[f"delta_{name}_all"] = sa[name] - peer[name]
                row[f"delta_{name}_both_completed"] = sa[name] - peer[name] if both else None
            paired.append(row)
    if len(paired) != 200 or len(first_divergences) != 200:
        raise RuntimeError("paired long-budget identity drift")
    cells = []
    for method in METHODS:
        for seed in ("7", "17", "29", "43", "61"):
            for split in ("regression", "frozen_check"):
                group = [r for r in episodes if (r["method"], r["seed"], r["split"]) == (method, seed, split)]
                if len(group) != (12 if split == "regression" else 8):
                    raise RuntimeError("per-seed/split episode count drift")
                for region in ("on_time", "late_completed", "incomplete"):
                    subset = [r for r in group if r["region"] == region]
                    cells.append({"method": method, "seed": seed, "split": split, "region": region,
                        "all_instances": len(group), "region_instances": len(subset),
                        "completed_coverage": sum(r["completed"] for r in group) / len(group),
                        "on_time_coverage": sum(r["on_time"] for r in group) / len(group),
                        **{f"mean_{name}_all": mean(r[name] for r in group) for name in ("modeled_completion_seconds", *COMPONENTS, "model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes")},
                        **{f"mean_{name}_region": mean(r[name] for r in subset) if subset else None for name in ("modeled_completion_seconds", *COMPONENTS, "model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes")},
                    })
    pair_cells = []
    for comparator in ("ppo", "dt_handoff_drl"):
        for split in ("regression", "frozen_check", "all"):
            group = [r for r in paired if r["comparator"] == comparator and (split == "all" or r["split"] == split)]
            both = [r for r in group if r["both_completed"]]
            pair_cells.append({"comparator": comparator, "split": split, "pairs": len(group),
                "sa_completed": sum(r["sa_completed"] for r in group), "peer_completed": sum(r["peer_completed"] for r in group),
                "sa_on_time": sum(r["sa_on_time"] for r in group), "peer_on_time": sum(r["peer_on_time"] for r in group),
                "both_completed": len(both), "both_completed_coverage": len(both) / len(group),
                **{f"mean_delta_{name}_all": mean(r[f"delta_{name}_all"] for r in group) for name in ("modeled_completion_seconds", *COMPONENTS, "model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes")},
                **{f"mean_delta_{name}_both_completed": mean(r[f"delta_{name}_both_completed"] for r in both) if both else None for name in ("modeled_completion_seconds", *COMPONENTS, "model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes")},
            })
    selection = []
    for method in METHODS:
        for seed in ("7", "17", "29", "43", "61"):
            group = [r for r in curve if r["method"] == method and r["seed"] == seed]
            if len(group) != 4 or sum(r["selected"] == "True" for r in group) != 1:
                raise RuntimeError("checkpoint selection curve identity drift")
            chosen = next(r for r in group if r["selected"] == "True")
            selection.append({"method": method, "seed": seed, "selected_update": int(chosen["update_index"]),
                "selected_dev_on_time": float(chosen["on_time_workflow_completion_rate"]),
                "selected_dev_completion": float(chosen["workflow_completion_rate"]),
                "selected_dev_elapsed": float(chosen["modeled_completion_seconds"]),
                "selected_dev_transfer_mb": float(chosen["total_transfer_mb"]),
                "candidate_dev_on_time": [float(r["on_time_workflow_completion_rate"]) for r in group],
                "candidate_updates": [int(r["update_index"]) for r in group]})
    return {"episodes": episodes, "paired_rows": paired, "first_divergences": first_divergences,
            "seed_split_region": cells, "paired_summary": pair_cells, "selection_summary": selection}


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"create-only diagnostic root exists: {OUTPUT}")
    config, splits, evaluation, ledger, curve, manifest = _verify_inputs()
    OUTPUT.mkdir(parents=True)
    _write_json(OUTPUT / "identity.json", {"scientific_commit": SCIENTIFIC_COMMIT,
        "run_id": SOURCE.name, "plan_sha256": _sha256(PLAN), "script_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(SOURCE / "run_manifest.json"),
        "postprocessed_selection_curve_sha256": _sha256(SOURCE / "checkpoint_selection_curve.csv"),
        "source_terminal_status": _load_json(TERMINAL)["status"],
        "source_scientific_completion": _load_json(SOURCE / "completion_receipt.json")["status"],
        "created_at": datetime.now(timezone.utc).isoformat(), "formal_or_holdout_reads": 0})
    events, episodes = _replay(config, splits, evaluation, ledger)
    summary = _aggregate(events, episodes, curve)
    with (OUTPUT / "event_rows.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    _write_json(OUTPUT / "episode_cost_rows.json", episodes)
    _write_json(OUTPUT / "paired_rows.json", summary["paired_rows"])
    _write_json(OUTPUT / "first_divergences.json", summary["first_divergences"])
    _write_csv(OUTPUT / "seed_split_region_cost.csv", summary["seed_split_region"])
    _write_csv(OUTPUT / "paired_cost_summary.csv", summary["paired_summary"])
    _write_json(OUTPUT / "checkpoint_selection_summary.json", summary["selection_summary"])
    _write_json(OUTPUT / "machine_summary.json", {"schema_version": "cscwd_sa_long_budget_cost_diagnosis_v1",
        "source_run_id": SOURCE.name, "scientific_commit": SCIENTIFIC_COMMIT, "event_rows": len(events),
        "episode_rows": len(episodes), "paired_rows": len(summary["paired_rows"]),
        "common_state_forward_calls": 0, "local_branch_states": 0, "training_steps": 0,
        "seed_split_region": summary["seed_split_region"], "paired_summary": summary["paired_summary"],
        "selection_summary": summary["selection_summary"],
        "claim_boundary": "consumed_development_only_not_paper_table_no_same_state_causal_claim_after_divergence"})
    _write_json(OUTPUT / "completion_receipt.json", {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "replayed_episodes": len(episodes), "replayed_steps": len(events), "paired_rows": len(summary["paired_rows"]),
        "common_state_forward_calls": 0, "local_branch_states": 0,
        "training_steps": 0, "formal_or_holdout_reads": 0})
    files = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != "artifact_integrity.json")
    _write_json(OUTPUT / "artifact_integrity.json", {"files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise RuntimeError("select exactly one of --preflight or --run")
    if args.preflight:
        _, _, evaluation, ledger, curve, _ = _verify_inputs()
        print(json.dumps({"status": "preflight_only", "episodes": len(evaluation), "behavior_rows": len(ledger), "selection_rows": len(curve), "new_evaluation": 0}))
    else:
        summary = run()
        print(json.dumps({"status": "complete", "output": str(OUTPUT), "replayed_episodes": len(summary["episodes"]), "paired_rows": len(summary["paired_rows"])}))


if __name__ == "__main__":
    main()
