"""Read-only, event-level reconstruction of the frozen phase-v2 development pilot."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


RUN_ID = "cscwd_phase_v2_causal_public_pilot_20261011_v1"
RUN_COMMIT = "5a79ffd018d4c260bc7a8417229a440c7d591fe4"
WORKLOAD_SHA = "b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc"
MATRIX_AUDIT_SHA = "30d821d4fb7c2e8d26c85378e4824fa4a782f62bf6eb1181510f0fe9bf56db8d"
SOURCE_SHA = {
    "completion_receipt.json": "88c2ac2c1cd7475b9adb3481d07d70cabe5657b4e0ec3c52fca6eb3fc2b0c8c8",
    "run_manifest.json": "8750250575566d68bc4c3030994536da6211db9d8aa65a6ce1a3a2173b84ea28",
    "artifact_integrity.json": "1ecea21b8daddbd79c908edf40a5adfbe38b8e411e9e30ffbc48c8974b95acb2",
    "evaluation_rows.csv": "126db6fa0ba9c0c58627bdacbc0ecc64e30f673fe934aabbe1c0c128504071f8",
}
TERMINAL_SHA = "2fbf478e3286cc6f8bf8fc031f81ca17904c19f23df60226329f3765d3879a1e"
V1 = "raw_ngsim_event_time_v1"
V2 = "raw_ngsim_event_time_v2"
LEARNED_ARMS = ("sa_control", "sa_candidate", "ppo", "mappo")
VIEWS = ("selected", "fixed_update24")
SEEDS = (7, 17, 29, 43, 61)
OBJECTIVE = {
    "node_completion_reward": 2.0,
    "workflow_completion_reward": 4.0,
    "time_weight": 0.08,
    "transfer_gib_weight": 0.5,
    "failure_penalty": 3.0,
    "deadline_penalty": 4.0,
}
DEADLINE_SECONDS = 99.164509
EPS = 1e-6


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value: Any) -> float:
    return float(value)


def integer(value: Any) -> int:
    return int(value)


def flag(value: Any) -> bool:
    if value in (True, "True", "true", 1, "1"):
        return True
    if value in (False, "False", "false", 0, "0"):
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def near(actual: float, expected: float, label: str, *, tolerance: float = EPS) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"{label}: actual {actual} != expected {expected}")


def verify_source(run_root: Path, terminal_path: Path) -> dict[str, Any]:
    if run_root.name != RUN_ID:
        raise RuntimeError("pilot run id differs from frozen identity")
    for name, expected in SOURCE_SHA.items():
        if sha(run_root / name) != expected:
            raise RuntimeError(f"frozen pilot artifact hash mismatch: {name}")
    if sha(terminal_path) != TERMINAL_SHA:
        raise RuntimeError("frozen supervisor terminal receipt hash mismatch")
    integrity = json.loads((run_root / "artifact_integrity.json").read_text())
    recorded = set()
    for item in integrity["files"]:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative in recorded:
            raise RuntimeError("invalid or duplicate integrity file path")
        recorded.add(relative)
        path = run_root / relative
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha(path) != item["sha256"]:
            raise RuntimeError(f"integrity mismatch: {relative}")
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    completion = json.loads((run_root / "completion_receipt.json").read_text())
    terminal = json.loads(terminal_path.read_text())
    if (terminal["status"] != "FAIL"
            or terminal["classification"] != "CHILD_EXECUTION_FAILURE_NO_RETRY"
            or terminal["return_code"] != 0
            or "scientific_execution_complete" in dict(terminal.get("child_receipt") or {})):
        raise RuntimeError("unexpected supervisor terminal/child receipt contract")
    if manifest["status"] != "complete" or manifest["git_commit"] != RUN_COMMIT:
        raise RuntimeError("pilot manifest not complete at frozen commit")
    if completion["status"] != "complete" or completion["git_commit"] != RUN_COMMIT:
        raise RuntimeError("pilot child completion not complete at frozen commit")
    counts = manifest["counts"]
    expected_counts = {
        "learned_cells": 20, "training_steps": 23040, "optimizer_steps": 3840,
        "checkpoint_selection_episodes": 80, "final_learned_evaluation_episodes": 320,
        "rule_evaluation_episodes": 18,
    }
    if any(int(counts[key]) != value for key, value in expected_counts.items()):
        raise RuntimeError("pilot scientific budget/count drift")
    if manifest["formal_or_holdout_reads"] != 0 or completion["formal_or_holdout_reads"] != 0:
        raise RuntimeError("formal/holdout read detected")
    if manifest["reward_profile"] != "original_reward_v1" or manifest["critic_target_normalization"] != "raw_disabled":
        raise RuntimeError("frozen reward or critic contract drift")
    gate = manifest["gate"]
    if (gate["status"] != "PASS" or gate["git_commit"] != RUN_COMMIT
            or gate["input_sha256"]["workload_manifest"] != WORKLOAD_SHA
            or gate["input_sha256"]["mechanism_matrix_audit"] != MATRIX_AUDIT_SHA
            or gate["mechanism_matrix_gate"]["status"] != "PASS"):
        raise RuntimeError("frozen matrix or source gate identity drift")
    return {
        "integrity_file_count": len(recorded),
        "child_completion": completion,
        "supervisor_terminal_status": terminal["status"],
        "supervisor_terminal_classification": terminal["classification"],
        "supervisor_return_code": terminal["return_code"],
        "supervisor_child_receipt_has_scientific_execution_complete": (
            "scientific_execution_complete" in dict(terminal.get("child_receipt") or {})
        ),
        "manifest_gate": gate,
        "manifest_counts": counts,
    }


def episode_key(row: dict[str, str], *, rule: bool) -> tuple[str, ...]:
    actor = row["method"] if rule else row["arm"]
    return (
        actor, row["view"], row["seed"], row["time_profile"],
        row["condition_id"], row["source_segment_id"], row["window_id"],
    )


def phase_events(step: dict[str, str]) -> list[dict[str, Any]]:
    events = json.loads(step["phase_events"])
    if not isinstance(events, list):
        raise RuntimeError("phase_events is not a list")
    return events


def recompute_episode(row: dict[str, str], steps: list[dict[str, str]]) -> dict[str, Any]:
    if not steps:
        raise RuntimeError("empty evaluation behavior episode")
    steps = sorted(steps, key=lambda item: integer(item["step"]))
    if [integer(item["step"]) for item in steps] != list(range(len(steps))):
        raise RuntimeError("noncontiguous evaluation step indices")
    if len(steps) > 24:
        raise RuntimeError("evaluation step cap exceeded")
    profile = row["time_profile"]
    counts = Counter(integer(step["executed_action"]) for step in steps)
    if any(integer(step["requested_action"]) != integer(step["executed_action"]) for step in steps):
        raise RuntimeError("evaluation requested/executed action mismatch")
    if any(not json.loads(step["action_mask"])[integer(step["executed_action"])] for step in steps):
        raise RuntimeError("masked evaluation action")
    completed_nodes = sum(flag(step["service_completed"]) for step in steps)
    service_failures = len(steps) - completed_nodes
    migration_successes = sum(flag(step["migration_success"]) for step in steps)
    terminal = flag(steps[-1]["terminated"])
    raw_cost = sum(number(step["step_cost_seconds"]) for step in steps)
    # The frozen raw profiles start after one observed 100-ms sample.
    elapsed = 0.1 + raw_cost
    total_reward = sum(number(step["reward"]) for step in steps)
    transfer = {
        key: sum(integer(step[key]) for step in steps)
        for key in ("input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
                    "backhaul_transfer_bytes", "partial_transfer_bytes")
    }
    total_bytes = transfer["input_transfer_bytes"] + transfer["model_transfer_bytes"] + transfer["state_transfer_bytes"]
    if profile == V2:
        if transfer["backhaul_transfer_bytes"] != transfer["model_transfer_bytes"] + transfer["state_transfer_bytes"]:
            raise RuntimeError("v2 backhaul bytes do not conserve")
    elif profile == V1:
        if transfer["backhaul_transfer_bytes"] or transfer["partial_transfer_bytes"]:
            raise RuntimeError("v1 reports v2-only transfer fields")
    else:
        raise RuntimeError("unknown physical profile")
    phases: defaultdict[str, float] = defaultdict(float)
    per_event = []
    for step in steps:
        parsed = phase_events(step)
        if profile == V2:
            near(sum(number(event["elapsed_seconds"]) for event in parsed),
                 number(step["step_cost_seconds"]), "v2 per-step phase seconds")
            phase_bytes = Counter()
            for event in parsed:
                if number(event["elapsed_seconds"]) < -EPS or integer(event["transferred_bytes"]) < 0:
                    raise RuntimeError("negative phase cost")
                phases[str(event["phase"])] += number(event["elapsed_seconds"])
                phase_bytes[str(event["phase"])] += integer(event["transferred_bytes"])
            near(phase_bytes["input_upload"] + phase_bytes["vehicle_fallback_service"],
                 integer(step["input_transfer_bytes"]), "v2 input phase bytes")
            near(phase_bytes["model_prepare_network"], integer(step["model_transfer_bytes"]),
                 "v2 model phase bytes")
            near(phase_bytes["state_transfer"], integer(step["state_transfer_bytes"]),
                 "v2 state phase bytes")
        elif parsed:
            raise RuntimeError("v1 exposed unexpected phase ledger")
        per_event.append({
            "step": integer(step["step"]), "action": integer(step["executed_action"]),
            "service_completed": flag(step["service_completed"]),
            "migration_success": flag(step["migration_success"]),
            "migration_status": step["migration_status"],
            "failure_reason": step["failure_reason"],
            "input_transfer_bytes": integer(step["input_transfer_bytes"]),
            "model_transfer_bytes": integer(step["model_transfer_bytes"]),
            "state_transfer_bytes": integer(step["state_transfer_bytes"]),
            "partial_transfer_bytes": integer(step["partial_transfer_bytes"]),
            "step_cost_seconds": number(step["step_cost_seconds"]),
            "reward": number(step["reward"]), "phases": parsed,
        })
    for action in range(5):
        if counts[action] != integer(row[f"action_{action}"]):
            raise RuntimeError("action histogram differs from behavior rows")
    if completed_nodes != integer(row["completed_nodes"]) or completed_nodes > integer(row["node_count"]):
        raise RuntimeError("completed node count mismatch")
    if service_failures != integer(row["service_failures"]):
        raise RuntimeError("service failure count mismatch")
    if migration_successes != integer(row["migration_successes"]):
        raise RuntimeError("migration success count mismatch")
    if profile == V2 and counts[4] != integer(row["migration_attempts"]):
        raise RuntimeError("v2 migration attempt count mismatch")
    for key, value in transfer.items():
        if value != integer(row[key]):
            raise RuntimeError(f"episode transfer mismatch: {key}")
    near(total_bytes / 1_000_000.0, number(row["total_transfer_mb"]), "total MB")
    near(float(terminal), number(row["workflow_completion_rate"]), "workflow completion")
    near(float(terminal and elapsed <= DEADLINE_SECONDS + EPS),
         number(row["on_time_workflow_completion_rate"]), "on-time completion")
    near(float(terminal and elapsed > DEADLINE_SECONDS + EPS),
         number(row["late_workflow_completion_rate"]), "late completion")
    near(float(not terminal), number(row["unfinished_rate"]), "unfinished")
    near(float(service_failures > 0), number(row["service_failure_rate"]),
         "binary episode service-failure rate")
    if terminal:
        near(elapsed, number(row["completed_sample_elapsed_seconds"]), "completed elapsed")
        if row["elapsed_seconds_for_failed_or_truncated"]:
            raise RuntimeError("completed episode reports failed elapsed")
    else:
        near(elapsed, number(row["elapsed_seconds_for_failed_or_truncated"]), "failed elapsed")
        if row["completed_sample_elapsed_seconds"]:
            raise RuntimeError("unfinished episode reports completed elapsed")
    if profile == V2:
        near(phases.get("prefix_recompute", 0.0), number(row["recompute_seconds"]), "v2 recompute")
    for name, seconds in json.loads(row["phase_elapsed_seconds"]).items():
        near(phases.get(name, 0.0), number(seconds), f"phase total {name}")
    if profile == V2 and set(phases) != set(json.loads(row["phase_elapsed_seconds"])):
        raise RuntimeError("v2 phase keys mismatch")
    near(total_reward, number(row["reward"]), "step reward sum")
    reward_parts = {
        "reward_node_completion": OBJECTIVE["node_completion_reward"] * completed_nodes,
        "reward_workflow_completion": OBJECTIVE["workflow_completion_reward"] * int(terminal),
        "reward_time_penalty": -OBJECTIVE["time_weight"] * raw_cost,
        "reward_transfer_penalty": -OBJECTIVE["transfer_gib_weight"] * total_bytes / (1024**3),
        "reward_failure_penalty": -OBJECTIVE["failure_penalty"] * service_failures,
        "reward_deadline_penalty": -OBJECTIVE["deadline_penalty"] * int(terminal and elapsed > DEADLINE_SECONDS + EPS),
    }
    for name, expected in reward_parts.items():
        near(number(row[name]), expected, f"reward component {name}")
    near(sum(reward_parts.values()), total_reward, "original_reward_v1 formula")
    return {
        "elapsed_seconds": elapsed, "workflow_completed": int(terminal),
        "on_time": int(terminal and elapsed <= DEADLINE_SECONDS + EPS),
        "service_failure_episode": int(service_failures > 0),
        "service_failure_events": service_failures,
        "recompute_seconds": number(row["recompute_seconds"]),
        "reward": total_reward, "total_transfer_bytes": total_bytes,
        **transfer, "events": per_event,
    }


def annotate_events(key: tuple[str, ...], episode: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    events = episode["events"]
    annotated = []
    counts = Counter()
    for position, event in enumerate(events):
        phases = event["phases"]
        compute_locations = [str(phase["location"]) for phase in phases if phase["phase"] == "current_node_compute"]
        future_compute = {
            str(phase["location"])
            for later in events[position + 1:]
            for phase in later["phases"] if phase["phase"] == "current_node_compute"
        }
        model_phase = next((phase for phase in phases if phase["phase"] == "model_prepare_network"), None)
        model_target = (
            str(model_phase["location"]).removeprefix("repository_to_")
            if model_phase and integer(model_phase["transferred_bytes"]) > 0 else ""
        )
        model_use = "no_model_transfer"
        if model_target:
            if event["action"] == 0 and model_target in compute_locations and event["service_completed"]:
                model_use = "same_step_current_service_observed"
            elif model_target in future_compute:
                model_use = "later_compute_at_prepared_rsu_observed"
            else:
                model_use = "no_later_compute_at_prepared_rsu_observed"
            counts[f"model_use:{model_use}:bytes"] += event["model_transfer_bytes"]
            counts[f"model_use:{model_use}:events"] += 1
        state_phase = next((phase for phase in phases if phase["phase"] == "state_transfer"), None)
        state_target = ""
        post_delivery_use = "not_applicable"
        if state_phase and event["state_transfer_bytes"]:
            location = str(state_phase["location"])
            state_target = location.split("_to_")[-1] if "_to_" in location else ""
            if event["migration_success"]:
                subsequent = next((later for later in events[position + 1:]
                                   if any(phase["phase"] == "current_node_compute" and
                                          str(phase["location"]) == state_target
                                          for phase in later["phases"])), None)
                if subsequent is None:
                    post_delivery_use = "no_later_target_compute_observed"
                elif any(phase["phase"] == "prefix_recompute" for phase in subsequent["phases"]):
                    post_delivery_use = "later_target_compute_with_recompute"
                else:
                    post_delivery_use = "later_target_compute_without_recompute"
            else:
                post_delivery_use = "state_sent_without_successful_delivery"
            counts[f"state:{post_delivery_use}:bytes"] += event["state_transfer_bytes"]
            counts[f"state:{post_delivery_use}:events"] += 1
        recompute = sum(number(phase["elapsed_seconds"]) for phase in phases
                        if phase["phase"] == "prefix_recompute")
        counts["recompute_microseconds"] += round(recompute * 1e6)
        annotated.append({
            "actor": key[0], "view": key[1], "seed": key[2],
            "time_profile": key[3], "condition_id": key[4],
            "source_segment_id": key[5], "window_id": key[6],
            "step": event["step"], "executed_action": event["action"],
            "service_completed": event["service_completed"],
            "migration_success": event["migration_success"],
            "migration_status": event["migration_status"],
            "failure_reason": event["failure_reason"],
            "input_transfer_bytes": event["input_transfer_bytes"],
            "model_transfer_bytes": event["model_transfer_bytes"],
            "state_transfer_bytes": event["state_transfer_bytes"],
            "partial_transfer_bytes": event["partial_transfer_bytes"],
            "recompute_seconds": recompute,
            "step_cost_seconds": event["step_cost_seconds"],
            "reward": event["reward"],
            "model_prepare_rsu_id": model_target,
            "model_observed_use": model_use,
            "state_transfer_target_rsu_id": state_target,
            "post_delivery_observation": post_delivery_use,
        })
    return annotated, counts


def summarize_rows(rows: list[dict[str, str]], *, rule: bool) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        actor = row["method"] if rule else row["arm"]
        key = ("privileged_reference" if rule and actor.startswith("privileged_") else
               "public_rule" if rule else "learned", row["view"], row["time_profile"],
               row["condition_id"], row["source_segment_id"], actor)
        groups[key].append(row)
    metrics = (
        "workflow_completion_rate", "on_time_workflow_completion_rate",
        "service_failure_rate", "service_failures", "total_transfer_mb",
        "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
        "backhaul_transfer_bytes", "partial_transfer_bytes", "recompute_seconds", "reward",
        "migration_attempts", "migration_successes", "state_ready_count", "decision_overhead_ms",
    )
    output = []
    for key, subset in sorted(groups.items()):
        if (len(subset) != 1 if rule else len(subset) != 5):
            raise RuntimeError("unexpected rule or seed count within stratum")
        elapsed_values = [number(row["completed_sample_elapsed_seconds"])
                          for row in subset if row["completed_sample_elapsed_seconds"]]
        output.append({
            "family": key[0], "view": key[1], "time_profile": key[2],
            "condition_id": key[3], "source_segment_id": key[4], "actor": key[5],
            "episode_count": len(subset), "unique_seed_count": len({row["seed"] for row in subset}),
            "independent_source_count": 1,
            "completed_elapsed_denominator": len(elapsed_values),
            "completed_elapsed_mean_seconds": statistics.mean(elapsed_values) if elapsed_values else "",
            **{f"{metric}_mean": statistics.mean(number(row[metric]) for row in subset)
               for metric in metrics},
        })
    return output


def paired_sa(evaluation: list[dict[str, str]], episodes: dict[tuple[str, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    controls = {episode_key(row, rule=False)[1:]: row for row in evaluation if row["arm"] == "sa_control"}
    candidates = {episode_key(row, rule=False)[1:]: row for row in evaluation if row["arm"] == "sa_candidate"}
    if controls.keys() != candidates.keys() or len(controls) != 80:
        raise RuntimeError("SA pair grid incomplete")
    metrics = (
        "workflow_completion_rate", "on_time_workflow_completion_rate", "service_failure_rate",
        "service_failures", "total_transfer_mb", "input_transfer_bytes", "model_transfer_bytes",
        "state_transfer_bytes", "backhaul_transfer_bytes", "partial_transfer_bytes",
        "recompute_seconds", "reward", "completed_sample_elapsed_seconds",
    )
    result = []
    for suffix in sorted(controls):
        control, candidate = controls[suffix], candidates[suffix]
        c_events = episodes[("sa_control",) + suffix]["events"]
        n_events = episodes[("sa_candidate",) + suffix]["events"]
        divergence = next((index for index in range(min(len(c_events), len(n_events)))
                           if c_events[index]["action"] != n_events[index]["action"]), None)
        row = {
            "view": suffix[0], "seed": suffix[1], "time_profile": suffix[2],
            "condition_id": suffix[3], "source_segment_id": suffix[4], "window_id": suffix[5],
            "first_action_divergence_step": divergence if divergence is not None else "",
            "same_action_sequence": [item["action"] for item in c_events] == [item["action"] for item in n_events],
            "control_steps": len(c_events), "candidate_steps": len(n_events),
            **{f"{metric}_candidate_minus_control": number(candidate[metric]) - number(control[metric])
               for metric in metrics},
        }
        result.append(row)
    return result


def verify_training_cells(
    signals: list[dict[str, str]], updates: list[dict[str, str]],
    optimizer: list[dict[str, str]], summaries: list[dict[str, Any]],
) -> None:
    expected = {(row["arm"], integer(row["seed"])) for row in summaries}
    if len(expected) != 20 or {arm for arm, _ in expected} != set(LEARNED_ARMS):
        raise RuntimeError("training cell identity mismatch")
    for rows, expected_per_cell, name in (
        (signals, 1152, "signal"), (updates, 24, "update"),
        (optimizer, 192, "optimizer"),
    ):
        counts = Counter((row["arm"], integer(row["seed"])) for row in rows)
        if set(counts) != expected or any(value != expected_per_cell for value in counts.values()):
            raise RuntimeError(f"{name} budget per cell mismatch")
    update_indices: defaultdict[tuple[str, int], set[int]] = defaultdict(set)
    for row in updates:
        key = (row["arm"], integer(row["seed"]))
        index = integer(row["update_index"])
        update_indices[key].add(index)
        if integer(row["environment_steps"]) != 48 * index:
            raise RuntimeError("update cumulative environment step mismatch")
    if any(indices != set(range(1, 25)) for indices in update_indices.values()):
        raise RuntimeError("update index grid mismatch")
    optimizer_counts = Counter(
        (row["arm"], integer(row["seed"]), integer(row["update_index"])) for row in optimizer
    )
    if len(optimizer_counts) != 20 * 24 or any(value != 8 for value in optimizer_counts.values()):
        raise RuntimeError("optimizer steps per update mismatch")
    signal_steps: defaultdict[tuple[str, int], set[int]] = defaultdict(set)
    for row in signals:
        signal_steps[(row["arm"], integer(row["seed"]))].add(integer(row["global_step"]))
    if any(steps != set(range(1, 1153)) for steps in signal_steps.values()):
        raise RuntimeError("training signal global step grid mismatch")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refuse empty analysis CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_audit(run_root: Path, terminal_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    source = verify_source(run_root, terminal_path)
    evaluation = load_csv(run_root / "evaluation_rows.csv")
    rule_evaluation = load_csv(run_root / "rule_evaluation_rows.csv")
    behavior = load_csv(run_root / "behavior_rows.csv")
    rule_behavior = load_csv(run_root / "rule_behavior_rows.csv")
    if (len(evaluation), len(rule_evaluation), len(behavior), len(rule_behavior)) != (320, 18, 2760, 144):
        raise RuntimeError("evaluation/behavior row counts differ from frozen run")
    training_signal = load_csv(run_root / "training_signal_rows.csv")
    training_updates = load_csv(run_root / "training_updates.csv")
    optimizer_rows = load_csv(run_root / "optimizer_rows.csv")
    if (len(training_signal), len(training_updates), len(optimizer_rows)) != (23040, 480, 3840):
        raise RuntimeError("training/optimizer row counts differ from frozen run")
    candidates = json.loads((run_root / "checkpoint_candidates.json").read_text())
    summaries = json.loads((run_root / "training_summary.json").read_text())
    if len(candidates) != 80 or len(summaries) != 20:
        raise RuntimeError("checkpoint/learned cell count mismatch")
    verify_training_cells(training_signal, training_updates, optimizer_rows, summaries)
    selected_updates = Counter()
    grouped_candidates: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if (candidate["development_row"]["source_segment_id"] != "us_101"
                or candidate["development_row"]["condition_id"] != "default"
                or candidate["development_row"]["time_profile"] != V2
                or candidate["reward_used_for_selection"] is not False
                or candidate["final_evaluation_used_for_selection"] is not False):
            raise RuntimeError("checkpoint selection provenance drift")
        dev = candidate["development_row"]
        score = [number(dev["on_time_workflow_completion_rate"]), number(dev["workflow_completion_rate"]),
                 -number(dev["service_failure_rate"]), -number(dev["unfinished_rate"]),
                 -number(dev["total_transfer_mb"]), -number(candidate["update_index"])]
        if score != candidate["score"]:
            raise RuntimeError("checkpoint selection score mismatch")
        if sha(run_root / candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
            raise RuntimeError("candidate checkpoint hash mismatch")
        grouped_candidates[(candidate["arm"], integer(candidate["seed"]))].append(candidate)
    for summary in summaries:
        key = (summary["arm"], integer(summary["seed"]))
        rows = grouped_candidates[key]
        if len(rows) != 4 or {integer(row["update_index"]) for row in rows} != {6, 12, 18, 24}:
            raise RuntimeError("checkpoint candidate grid incomplete")
        best = max(rows, key=lambda row: tuple(row["score"]))
        if integer(best["update_index"]) != integer(summary["selected_update"]):
            raise RuntimeError("selected checkpoint update mismatch")
        if summary["environment_steps"] != 1152 or summary["updates"] != 24 or summary["optimizer_steps"] != 192:
            raise RuntimeError("cell training budget mismatch")
        if sha(run_root / summary["selected_checkpoint"]) != summary["selected_checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint hash mismatch")
        if sha(run_root / summary["fixed_update24_checkpoint"]) != summary["fixed_update24_checkpoint_sha256"]:
            raise RuntimeError("fixed checkpoint hash mismatch")
        selected_updates[integer(summary["selected_update"])] += 1
    for rows, ledger, rule in ((evaluation, behavior, False), (rule_evaluation, rule_behavior, True)):
        if len({episode_key(row, rule=rule) for row in rows}) != len(rows):
            raise RuntimeError("duplicate evaluation episode identity")
        if set(episode_key(row, rule=rule) for row in rows) != set(episode_key(row, rule=rule) for row in ledger):
            raise RuntimeError("evaluation/behavior episode identity mismatch")
    behavior_by_key: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in behavior:
        behavior_by_key[episode_key(row, rule=False)].append(row)
    rule_behavior_by_key: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rule_behavior:
        rule_behavior_by_key[episode_key(row, rule=True)].append(row)
    episodes = {}
    event_rows = []
    full_method_rows = []
    attribution = Counter()
    action_sequences: defaultdict[tuple[str, ...], set[tuple[int, ...]]] = defaultdict(set)
    for rows, grouped, rule in ((evaluation, behavior_by_key, False), (rule_evaluation, rule_behavior_by_key, True)):
        for row in rows:
            key = episode_key(row, rule=rule)
            episode = recompute_episode(row, grouped[key])
            episodes[key] = episode
            family = ("privileged_reference" if rule and row["arm"] == "privileged_rule"
                      else "public_rule" if rule else "learned")
            full_method_rows.append({
                "family": family, "actor": key[0], "method": row["method"],
                "capability_profile": row.get("capability_profile", "learned_shared_public_state"),
                "view": key[1], "seed": key[2], "time_profile": key[3],
                "condition_id": key[4], "source_segment_id": key[5], "window_id": key[6],
                "workflow_completed": episode["workflow_completed"],
                "on_time": episode["on_time"],
                "service_failure_episode": episode["service_failure_episode"],
                "service_failure_events": episode["service_failure_events"],
                "elapsed_seconds": episode["elapsed_seconds"],
                "total_transfer_mb": episode["total_transfer_bytes"] / 1e6,
                "input_transfer_bytes": episode["input_transfer_bytes"],
                "model_transfer_bytes": episode["model_transfer_bytes"],
                "state_transfer_bytes": episode["state_transfer_bytes"],
                "backhaul_transfer_bytes": episode["backhaul_transfer_bytes"],
                "partial_transfer_bytes": episode["partial_transfer_bytes"],
                "recompute_seconds": episode["recompute_seconds"],
                "reward": episode["reward"],
                "migration_attempts": integer(row["migration_attempts"]),
                "migration_successes": integer(row["migration_successes"]),
                "state_ready_count_reported": integer(row["state_ready_count"]),
                "decision_overhead_ms_reported": number(row["decision_overhead_ms"]),
                "steps": len(episode["events"]),
                **{f"action_{action}": sum(event["action"] == action for event in episode["events"])
                   for action in range(5)},
            })
            if not rule:
                sequence_key = (row["arm"], row["view"], row["time_profile"],
                                row["condition_id"], row["source_segment_id"])
                action_sequences[sequence_key].add(tuple(event["action"] for event in episode["events"]))
            annotated, counts = annotate_events(key, episode)
            event_rows.extend(annotated)
            if not rule and row["time_profile"] == V2 and row["view"] == "selected":
                for name, value in counts.items():
                    attribution[(row["arm"], name)] += value
    public_rules = [row for row in rule_evaluation if row["arm"] == "public_rule" and row["time_profile"] == V2]
    for row in public_rules:
        if row.get("phase_v2_independent_baseline") != "False":
            raise RuntimeError("v2 public rule misclassified as independent")
    by_rule = {(row["source_segment_id"], row["condition_id"], row["method"]): row for row in public_rules}
    for source_segment in ("lankershim", "us_101"):
        for condition in ("default", "adverse_link", "ample_cache"):
            immediate = by_rule[(source_segment, condition, "causal_public_immediate_rule")]
            two_step = by_rule[(source_segment, condition, "causal_public_two_step_rule")]
            a = episodes[episode_key(immediate, rule=True)]["events"]
            b = episodes[episode_key(two_step, rule=True)]["events"]
            if [event["action"] for event in a] != [event["action"] for event in b]:
                raise RuntimeError("v2 two-step did not fail closed to immediate action sequence")
    privileged = [row for row in rule_evaluation if row["arm"] == "privileged_rule"]
    if len(privileged) != 2 or any(row["time_profile"] != V1 or row["phase_v2_available"] != "False"
                                  for row in privileged):
        raise RuntimeError("privileged capability boundary drift")
    stratified = summarize_rows(evaluation, rule=False) + summarize_rows(rule_evaluation, rule=True)
    pairs = paired_sa(evaluation, episodes)
    output_root.mkdir(parents=True)
    full_path = output_root / "full_method_rows.csv"
    stratified_path = output_root / "stratified_methods.csv"
    event_path = output_root / "event_attribution.csv"
    pair_path = output_root / "paired_sa_cases.csv"
    write_csv(full_path, full_method_rows)
    write_csv(stratified_path, stratified)
    write_csv(event_path, event_rows)
    write_csv(pair_path, pairs)
    summary = {
        "schema_version": "cscwd_phase_v2_pilot_independent_audit_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": RUN_ID, "source_git_commit": RUN_COMMIT,
        "source_sha256": SOURCE_SHA,
        "supervisor_terminal_sha256": TERMINAL_SHA,
        "source_integrity_file_count": source["integrity_file_count"],
        "child_completion_status": source["child_completion"]["status"],
        "supervisor_terminal_status": source["supervisor_terminal_status"],
        "supervisor_terminal_classification": source["supervisor_terminal_classification"],
        "supervisor_return_code": source["supervisor_return_code"],
        "supervisor_schema_mismatch": not source["supervisor_child_receipt_has_scientific_execution_complete"],
        "learned_evaluation_episodes": len(evaluation),
        "rule_evaluation_episodes": len(rule_evaluation),
        "learned_behavior_steps": len(behavior), "rule_behavior_steps": len(rule_behavior),
        "training_steps": len(training_signal), "training_update_rows": len(training_updates),
        "optimizer_steps": len(optimizer_rows),
        "checkpoint_candidates": len(candidates), "selected_checkpoint_updates": dict(sorted(selected_updates.items())),
        "full_method_rows": len(full_method_rows),
        "stratified_method_rows": len(stratified), "paired_sa_cases": len(pairs),
        "unique_learned_action_sequences_by_stratum": [
            {"arm": key[0], "view": key[1], "time_profile": key[2],
             "condition_id": key[3], "source_segment_id": key[4],
             "unique_action_sequences_across_five_seeds": len(sequences)}
            for key, sequences in sorted(action_sequences.items())
        ],
        "event_attribution_counts_selected_v2": {
            f"{actor}:{name}": value for (actor, name), value in sorted(attribution.items())
        },
        "science_boundary": {
            "development_only": True, "training_source_from_protocol_and_runner": "lankershim",
            "checkpoint_selection_source_from_candidates": "us_101",
            "final_evaluation_reuses_training_and_selection_sources": True,
            "independent_source_count": 2,
            "five_seeds_and_two_views_are_repeated_within_sources": True,
            "phase_v2_two_step_independent_baseline": False,
            "privileged_reference_v2_available": False,
            "formal_or_holdout_results_read": False,
            "paper_ready": False,
        },
        "output_sha256": {
            "full_method_rows.csv": sha(full_path),
            "stratified_methods.csv": sha(stratified_path),
            "event_attribution.csv": sha(event_path),
            "paired_sa_cases.csv": sha(pair_path),
        },
    }
    summary_path = output_root / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--terminal-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_audit(args.run_root.resolve(), args.terminal_receipt.resolve(), args.output_root.resolve())
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "summary_sha256": sha(args.output_root.resolve() / "audit_summary.json"),
        "learned_evaluation_episodes": result["learned_evaluation_episodes"],
        "supervisor_terminal_status": result["supervisor_terminal_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
