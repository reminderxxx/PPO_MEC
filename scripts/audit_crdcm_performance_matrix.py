#!/usr/bin/env python3
"""Independently audit and re-aggregate the completed CRDCM v2 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import torch


ROOT = Path(__file__).resolve().parents[1]
AUDIT_VERSION = "crdcm_performance_matrix_independent_audit_v1"
POLICY_VERSION = "tmc_review_policy_v3_20260621"
TARGET_VENUE = "IEEE Transactions on Mobile Computing (TMC)"
LITERATURE_CUTOFF = "2026-09-28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) in {"True", "true", "1"}:
        return True
    if str(value) in {"False", "false", "0", ""}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def optional_float(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def recursive_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            recursive_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def count_request_failures(summary: dict[str, Any]) -> dict[str, Any]:
    endpoint = dict(summary["formal_request_execution_audit"])
    exposure = dict(summary["formal_request_exposure"])
    events = [
        event
        for event in summary.get("cache_event_trace", [])
        if event.get("event_type") == "request"
    ]
    steps = list(summary.get("step_trace", []))
    if len(events) != len(steps):
        raise ValueError("request event and step trace lengths differ")
    failed = [
        (index, event, steps[index])
        for index, event in enumerate(events)
        if not bool(event.get("service_success"))
    ]
    completed = bool(endpoint["workflow_completed_under_exogenous_execution"])
    right_censored = bool(endpoint["right_censored"])
    handoff_unready = [
        item
        for item in failed
        if int(item[2].get("handoff_event_count", 0) or 0) > 0
        and not bool(item[2].get("handoff_ready"))
    ]
    dependency_miss = [
        item
        for item in failed
        if not bool(item[1].get("base_model_hit", True))
        or not bool(item[1].get("adapter_hit", True))
        or not bool(item[1].get("joint_model_hit", True))
    ]
    state_not_ready = [
        item for item in failed if not bool(item[1].get("workflow_state_ready", True))
    ]
    invalid = [
        item
        for item in failed
        if bool(item[2].get("action_invalid"))
        or not bool(item[2].get("action_precondition_valid", True))
    ]
    stalls = [item for item in failed if bool(item[1].get("stall_occurred"))]
    capacity_rejections = [
        item for item in failed if item[1].get("capacity_rejection_reason")
    ]
    if completed:
        primary = "completed"
    elif right_censored:
        primary = "right_censored"
    elif not failed:
        primary = "incomplete_without_request_failure"
    elif handoff_unready:
        primary = "handoff_unprepared"
    elif dependency_miss:
        primary = "cache_dependency_miss"
    elif state_not_ready:
        primary = "workflow_state_unready"
    elif invalid:
        primary = "action_invalid_or_precondition"
    elif stalls:
        primary = "stall_or_timeout"
    else:
        primary = "other_request_failure"
    denominator = int(endpoint["external_request_denominator"])
    exposure_count = len(exposure["requests"])
    return {
        "primary": primary,
        "request_count": len(events),
        "request_success_count": sum(bool(event.get("service_success")) for event in events),
        "request_failure_count": len(failed),
        "dependency_miss_count": len(dependency_miss),
        "handoff_unprepared_count": len(handoff_unready),
        "capacity_rejection_count": len(capacity_rejections),
        "invalid_or_precondition_count": len(invalid),
        "stall_count": len(stalls),
        "denominator_consistent": (
            len(events) == denominator == exposure_count
            and endpoint.get("request_alignment_status") == "pass"
        ),
    }


def controller_name(run_info: dict[str, Any]) -> str:
    condition = str(run_info["condition_id"])
    if condition == "crdcm_critical_path_heuristic":
        return condition
    return f"{condition}/seed_{int(run_info['seed'])}"


def row_from_episode(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    run_info = dict(summary["run_info"])
    endpoint = dict(summary["formal_request_execution_audit"])
    system = dict(summary["system_metrics"])
    failures = count_request_failures(summary)
    trace = list(summary.get("policy_decision_trace_v2", []))
    steps = list(summary.get("step_trace", []))
    actions = [int(item["actual_executed_action"]) for item in trace]
    override_count = sum(bool(item.get("override_applied")) for item in trace)
    condition = str(run_info["condition_id"])
    seed = None if condition == "crdcm_critical_path_heuristic" else int(run_info["seed"])
    migration_transfer_bytes = sum(
        float(item.get("cache_event", {}).get("state_migration_size_mb", 0.0) or 0.0)
        for item in steps
    )
    realized_migration_count = sum(
        bool(item.get("migration_prepare_realized"))
        or bool(item.get("migration_during_handoff"))
        for item in steps
    )
    migration_suppressed_count = sum(bool(item.get("migration_suppressed")) for item in steps)
    return {
        "controller": controller_name(run_info),
        "condition_id": condition,
        "seed": seed,
        "scenario_id": str(run_info["scenario_id"]),
        "unit_id": str(run_info["unit_id"]),
        "window_id": str(run_info["window_id"]),
        "workflow_id": str(run_info["workflow_id"]),
        "workflow_completed": bool(endpoint["workflow_completed_under_exogenous_execution"]),
        "workflow_continuity_rate": float(endpoint["workflow_continuity_rate"]),
        "handoff_failure_rate": float(system["handoff_failure_rate"]),
        "failure_category": failures["primary"],
        "request_count": failures["request_count"],
        "request_success_count": failures["request_success_count"],
        "request_failure_count": failures["request_failure_count"],
        "dependency_miss_count": failures["dependency_miss_count"],
        "handoff_unprepared_count": failures["handoff_unprepared_count"],
        "capacity_rejection_count": failures["capacity_rejection_count"],
        "denominator_consistent": failures["denominator_consistent"],
        "transfer_mb_per_request": float(endpoint["transfer_mb_per_request"]),
        "backhaul_traffic_cost": float(system["backhaul_traffic_cost"]),
        "migration_overhead": float(system["adapter_state_migration_overhead"]),
        "state_migration_size_mb": migration_transfer_bytes,
        "realized_migration_count": realized_migration_count,
        "migration_suppressed_count": migration_suppressed_count,
        "end_to_end_workflow_delay": endpoint.get("end_to_end_workflow_delay"),
        "delay_available": endpoint.get("end_to_end_workflow_delay") is not None,
        "total_steps": len(steps),
        "override_count": override_count,
        "override_rate": round(override_count / max(len(trace), 1), 6),
        "action_sequence": "|".join(str(item) for item in actions),
        "action_count": len(actions),
        "request_exposure_fingerprint": str(endpoint["request_exposure_fingerprint"]),
        "outcome_fingerprint": str(endpoint["outcome_fingerprint"]),
        "raw_episode_path": str(path),
        "raw_episode_sha256": sha256(path),
    }


def rounded_mean(values: Iterable[float]) -> float:
    return round(fmean(values), 6)


def aggregate_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    delays = [float(row["end_to_end_workflow_delay"]) for row in group if row["delay_available"]]
    requests = sum(int(row["request_count"]) for row in group)
    return {
        "episode_count": len(group),
        "completed_count": sum(bool(row["workflow_completed"]) for row in group),
        "completion_rate": rounded_mean(float(bool(row["workflow_completed"])) for row in group),
        "continuity_episode_mean": rounded_mean(float(row["workflow_continuity_rate"]) for row in group),
        "continuity_request_weighted": round(
            sum(int(row["request_success_count"]) for row in group) / max(requests, 1), 6
        ),
        "handoff_failure_episode_mean": rounded_mean(float(row["handoff_failure_rate"]) for row in group),
        "request_count": requests,
        "request_failure_count": sum(int(row["request_failure_count"]) for row in group),
        "dependency_miss_count": sum(int(row["dependency_miss_count"]) for row in group),
        "handoff_unprepared_count": sum(int(row["handoff_unprepared_count"]) for row in group),
        "capacity_rejection_count": sum(int(row["capacity_rejection_count"]) for row in group),
        "transfer_mb_per_request_episode_mean": rounded_mean(
            float(row["transfer_mb_per_request"]) for row in group
        ),
        "transfer_mb_per_request_request_weighted": round(
            sum(float(row["backhaul_traffic_cost"]) for row in group) / max(requests, 1), 6
        ),
        "migration_overhead_episode_mean": rounded_mean(
            float(row["migration_overhead"]) for row in group
        ),
        "migration_overhead_per_request": round(
            sum(float(row["migration_overhead"]) for row in group) / max(requests, 1), 6
        ),
        "delay_available_count": len(delays),
        "delay_coverage_rate": round(len(delays) / len(group), 6),
        "conditional_delay_mean": rounded_mean(delays) if delays else None,
        "failure_categories": "|".join(
            f"{name}:{count}"
            for name, count in sorted(Counter(row["failure_category"] for row in group).items())
        ),
        "override_rate": round(
            sum(int(row["override_count"]) for row in group)
            / max(sum(int(row["action_count"]) for row in group), 1),
            6,
        ),
    }


def grouped_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        output.append(
            {
                **dict(zip(keys, identity)),
                **aggregate_group(groups[identity]),
            }
        )
    return output


def compare_original_rows(
    source_root: Path, derived: list[dict[str, Any]], checkpoint_hashes: dict[tuple[str, int], str]
) -> dict[str, Any]:
    with (source_root / "evaluation/paired_episode_rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        original = list(csv.DictReader(handle))
    original_index = {
        (row["controller"], row["scenario_id"], row["unit_id"]): row
        for row in original
    }
    mismatches: list[dict[str, Any]] = []
    for row in derived:
        key = (row["controller"], row["scenario_id"], row["unit_id"])
        observed = original_index.get(key)
        if observed is None:
            mismatches.append({"key": key, "field": "row", "reason": "missing"})
            continue
        expected_values: dict[str, Any] = {
            "workflow_completed": row["workflow_completed"],
            "workflow_continuity_rate": row["workflow_continuity_rate"],
            "handoff_failure_rate": row["handoff_failure_rate"],
            "failure_category": row["failure_category"],
            "request_failure_count": row["request_failure_count"],
            "transfer_mb_per_request": row["transfer_mb_per_request"],
            "migration_cost": row["migration_overhead"],
            "end_to_end_workflow_delay": row["end_to_end_workflow_delay"],
            "delay_available": row["delay_available"],
            "total_steps": row["total_steps"],
            "override_count": row["override_count"],
            "override_rate": row["override_rate"],
            "action_sequence": row["action_sequence"],
            "action_count": row["action_count"],
            "condition_id": row["condition_id"],
            "seed": row["seed"],
            "checkpoint_sha256": (
                None
                if row["seed"] is None
                else checkpoint_hashes[(str(row["condition_id"]), int(row["seed"]))]
            ),
        }
        for field, expected in expected_values.items():
            actual: Any = observed[field]
            if isinstance(expected, bool):
                equal = as_bool(actual) == expected
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                equal = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
            elif expected is None:
                equal = actual == ""
            else:
                equal = str(actual) == str(expected)
            if not equal:
                mismatches.append(
                    {"key": key, "field": field, "expected": expected, "actual": actual}
                )
    return {
        "source_row_count": len(original),
        "derived_row_count": len(derived),
        "identity_count": len(original_index),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "pass" if len(original) == len(derived) == len(original_index) and not mismatches else "fail",
    }


def compare_original_aggregate(
    source_root: Path, derived: list[dict[str, Any]]
) -> dict[str, Any]:
    original = load_json(source_root / "evaluation/aggregate.json")
    independent = grouped_rows(derived, ("controller", "scenario_id"))
    index = {(row["controller"], row["scenario_id"]): row for row in independent}
    mismatches: list[dict[str, Any]] = []
    mapping = {
        "episode_count": "episode_count",
        "completed_count": "completed_count",
        "completion_rate": "completion_rate",
        "continuity_rate_mean": "continuity_episode_mean",
        "handoff_failure_rate_mean": "handoff_failure_episode_mean",
        "request_failure_count": "request_failure_count",
        "failure_categories": "failure_categories",
        "transfer_mb_per_request_mean": "transfer_mb_per_request_episode_mean",
        "migration_cost_mean": "migration_overhead_episode_mean",
        "delay_available_count": "delay_available_count",
        "delay_coverage_rate": "delay_coverage_rate",
        "conditional_delay_mean": "conditional_delay_mean",
        "override_rate": "override_rate",
    }
    for row in original:
        key = (row["controller"], row["scenario_id"])
        expected = index[key]
        for source_field, derived_field in mapping.items():
            left = row[source_field]
            right = expected[derived_field]
            if isinstance(left, (int, float)) and not isinstance(left, bool) and right is not None:
                equal = math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
            else:
                equal = left == right
            if not equal:
                mismatches.append(
                    {
                        "key": key,
                        "field": source_field,
                        "source": left,
                        "derived": right,
                    }
                )
    return {
        "source_group_count": len(original),
        "derived_group_count": len(independent),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "status": "pass" if len(original) == len(independent) and not mismatches else "fail",
    }


def verify_integrity(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "artifact_integrity_manifest.json"
    manifest = load_json(manifest_path)
    failures = []
    listed_paths = set()
    for entry in manifest["files"]:
        relative = str(entry["path"])
        listed_paths.add(relative)
        path = source_root / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        observed_sha = sha256(path)
        if observed_size != int(entry["size_bytes"]) or observed_sha != str(entry["sha256"]):
            failures.append(
                {
                    "path": relative,
                    "reason": "identity_mismatch",
                    "expected_size": entry["size_bytes"],
                    "observed_size": observed_size,
                    "expected_sha256": entry["sha256"],
                    "observed_sha256": observed_sha,
                }
            )
    actual_paths = {
        str(path.relative_to(source_root))
        for path in source_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    extra = sorted(actual_paths - listed_paths)
    missing_from_inventory = sorted(listed_paths - actual_paths)
    status = (
        manifest.get("status") == "pass"
        and int(manifest["file_count"]) == len(manifest["files"])
        and len(actual_paths) == len(listed_paths)
        and not failures
        and not extra
        and not missing_from_inventory
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "declared_file_count": int(manifest["file_count"]),
        "observed_file_count_excluding_manifest": len(actual_paths),
        "failure_count": len(failures),
        "failures": failures,
        "extra_paths": extra,
        "missing_paths": missing_from_inventory,
        "status": "pass" if status else "fail",
    }


def verify_training_and_checkpoints(
    source_root: Path,
    command_manifest: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    evaluation_receipt: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int], str]]:
    command_index = {
        (str(row["condition_id"]), int(row["seed"])): row
        for row in command_manifest["commands"]
    }
    checkpoint_index = {
        (str(row["condition_id"]), int(row["seed"])): row
        for row in checkpoint_manifest["entries"]
    }
    rows = []
    total_episodes = 0
    total_steps = 0
    total_updates = 0
    failures = []
    hashes = {}
    for identity in sorted(command_index):
        command = command_index[identity]
        checkpoint_entry = checkpoint_index[identity]
        summary_path = Path(str(command["train_summary_path"]))
        summary = load_json(summary_path)
        latest = Path(str(checkpoint_entry["checkpoint_path"]))
        endpoint = latest.with_name("update_0016.pt")
        latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
        endpoint_payload = torch.load(endpoint, map_location="cpu", weights_only=False)
        latest_sha = sha256(latest)
        endpoint_sha = sha256(endpoint)
        evaluation_before = evaluation_receipt["checkpoint_hashes_before"][str(latest)]
        evaluation_after = evaluation_receipt["checkpoint_hashes_after"][str(latest)]
        parameter_identity = dict(summary["parameter_identity"])
        before = dict(parameter_identity["before_training"])
        after = dict(parameter_identity["after_training"])
        update_logs = list(summary["update_logs"])
        episode_files = sorted(summary_path.parent.glob("episodes/episode_*.summary.json"))
        row_failures = []
        checks = {
            "episode_count": int(summary["episodes"]) == 64 == len(episode_files),
            "update_count": int(summary["update_count"]) == 16 == len(update_logs),
            "checkpoint_schedule": list(summary["saved_checkpoint_update_indices"]) == [4, 8, 12, 16],
            "manifest_identity": int(checkpoint_entry["episodes"]) == 64 and int(checkpoint_entry["update_count"]) == 16,
            "parameter_changed": bool(parameter_identity["changed"]) and before["sha256"] != after["sha256"],
            "parameters_finite": bool(before["all_finite"]) and bool(after["all_finite"]),
            "latest_sha": latest_sha == checkpoint_entry["checkpoint_sha256"],
            "evaluation_hash_before_after": latest_sha == evaluation_before == evaluation_after,
            "payload_update_count": int(latest_payload["update_count"]) == int(endpoint_payload["update_count"]) == 16,
            "endpoint_logical_equivalence": recursive_equal(latest_payload, endpoint_payload),
        }
        for name, passed in checks.items():
            if not passed:
                row_failures.append(name)
                failures.append({"identity": identity, "check": name})
        steps = sum(int(item["collected_steps"]) for item in update_logs)
        total_episodes += int(summary["episodes"])
        total_steps += steps
        total_updates += int(summary["update_count"])
        hashes[identity] = latest_sha
        rows.append(
            {
                "condition_id": identity[0],
                "seed": identity[1],
                "episodes": int(summary["episodes"]),
                "actual_steps": steps,
                "update_count": int(summary["update_count"]),
                "before_parameter_sha256": before["sha256"],
                "after_parameter_sha256": after["sha256"],
                "parameter_changed": parameter_identity["changed"],
                "parameters_finite": bool(before["all_finite"]) and bool(after["all_finite"]),
                "latest_checkpoint_path": str(latest),
                "latest_checkpoint_size_bytes": latest.stat().st_size,
                "latest_checkpoint_sha256": latest_sha,
                "endpoint_checkpoint_path": str(endpoint),
                "endpoint_checkpoint_size_bytes": endpoint.stat().st_size,
                "endpoint_checkpoint_sha256": endpoint_sha,
                "byte_identical_to_update_0016": latest_sha == endpoint_sha,
                "logical_payload_equal_to_update_0016": recursive_equal(latest_payload, endpoint_payload),
                "evaluation_before_sha256": evaluation_before,
                "evaluation_after_sha256": evaluation_after,
                "last_actor_loss": update_logs[-1].get("actor_loss"),
                "last_value_loss": update_logs[-1].get("value_loss"),
                "last_policy_entropy": update_logs[-1].get("policy_entropy"),
                "last_approx_kl": update_logs[-1].get("approx_kl"),
                "last_clip_fraction": update_logs[-1].get("clip_fraction"),
                "status": "pass" if not row_failures else "fail",
                "failed_checks": "|".join(row_failures),
            }
        )
    summary = {
        "command_identity_count": len(command_index),
        "checkpoint_identity_count": len(checkpoint_index),
        "training_episodes": total_episodes,
        "actual_training_steps": total_steps,
        "training_updates": total_updates,
        "latest_endpoint_byte_identical_count": sum(row["byte_identical_to_update_0016"] for row in rows),
        "latest_endpoint_logically_equal_count": sum(row["logical_payload_equal_to_update_0016"] for row in rows),
        "failure_count": len(failures),
        "failures": failures,
        "status": "pass" if len(command_index) == len(checkpoint_index) == 12 and not failures else "fail",
    }
    return rows, summary, hashes


def normalized_request_schedule(summary: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "request_order",
        "step_index",
        "time_index",
        "vehicle_id",
        "workflow_id",
        "node_id",
        "required_base_model",
        "adapter_id",
        "object_id",
        "object_size_mb",
        "request_rsu_id",
        "current_service_rsu_id",
        "eligible_service_rsu_ids",
        "eligible_cache_target_rsu_ids",
    )
    return [
        {field: request.get(field) for field in fields}
        for request in summary["formal_request_exposure"]["requests"]
    ]


def pairwise_trace_audit(
    derived: list[dict[str, Any]], summaries: dict[tuple[str, int | None, str, str], dict[str, Any]]
) -> dict[str, Any]:
    row_index = {
        (row["condition_id"], row["seed"], row["scenario_id"], row["workflow_id"]): row
        for row in derived
    }
    learned_fields = (
        "workflow_completed",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "failure_category",
        "request_failure_count",
        "transfer_mb_per_request",
        "migration_overhead",
        "end_to_end_workflow_delay",
        "action_sequence",
    )
    sa_mappo_changes = Counter()
    sa_mappo_action_equal = 0
    sa_mappo_examples = []
    for seed in (1401, 1402, 1403):
        for scenario in sorted({row["scenario_id"] for row in derived}):
            for workflow in ("j_3", "j_8", "j_15"):
                left = row_index[("crdcm_full_sa", seed, scenario, workflow)]
                right = row_index[("crdcm_full_mappo", seed, scenario, workflow)]
                if left["action_sequence"] == right["action_sequence"]:
                    sa_mappo_action_equal += 1
                changed = [field for field in learned_fields if left[field] != right[field]]
                sa_mappo_changes.update(changed)
                if changed:
                    sa_mappo_examples.append(
                        {
                            "seed": seed,
                            "scenario_id": scenario,
                            "workflow_id": workflow,
                            "changed_fields": changed,
                            "sa_actions": left["action_sequence"],
                            "mappo_actions": right["action_sequence"],
                            "sa_failure": left["failure_category"],
                            "mappo_failure": right["failure_category"],
                            "sa_request_failures": left["request_failure_count"],
                            "mappo_request_failures": right["request_failure_count"],
                        }
                    )

    capacity_pairs = []
    exact_metric_action = 0
    capacity_rejections = 0
    capacity_evictions = 0
    shared_evictions = 0
    capacity_occupancy = []
    shared_occupancy = []
    schedules_equal = 0
    comparable = [
        row for row in derived if row["scenario_id"] == "capacity_competition"
    ]
    for left in comparable:
        right = row_index[
            (
                left["condition_id"],
                left["seed"],
                "shared_adapter_reuse_prepare",
                left["workflow_id"],
            )
        ]
        changed = [field for field in learned_fields if left[field] != right[field]]
        if not changed:
            exact_metric_action += 1
        left_summary = summaries[
            (left["condition_id"], left["seed"], left["scenario_id"], left["workflow_id"])
        ]
        right_summary = summaries[
            (right["condition_id"], right["seed"], right["scenario_id"], right["workflow_id"])
        ]
        schedule_equal = normalized_request_schedule(left_summary) == normalized_request_schedule(right_summary)
        schedules_equal += int(schedule_equal)
        capacity_rejections += int(left["capacity_rejection_count"]) + int(right["capacity_rejection_count"])
        for step in left_summary["step_trace"]:
            capacity_evictions += int(step.get("eviction_count", 0) or 0)
            if step.get("cache_occupancy_rate") is not None:
                capacity_occupancy.append(float(step["cache_occupancy_rate"]))
        for step in right_summary["step_trace"]:
            shared_evictions += int(step.get("eviction_count", 0) or 0)
            if step.get("cache_occupancy_rate") is not None:
                shared_occupancy.append(float(step["cache_occupancy_rate"]))
        capacity_pairs.append(
            {
                "controller": left["controller"],
                "workflow_id": left["workflow_id"],
                "normalized_request_schedule_equal": schedule_equal,
                "metric_action_changed_fields": changed,
                "capacity_exposure_fingerprint": left["request_exposure_fingerprint"],
                "shared_exposure_fingerprint": right["request_exposure_fingerprint"],
            }
        )

    low_reuse = [
        row for row in derived if row["scenario_id"] == "low_reuse_no_migration_negative"
    ]
    low_reuse_steps = []
    for row in low_reuse:
        summary = summaries[
            (row["condition_id"], row["seed"], row["scenario_id"], row["workflow_id"])
        ]
        for step in summary["step_trace"]:
            overhead = float(step.get("adapter_state_migration_overhead", 0.0) or 0.0)
            if overhead > 0.0:
                low_reuse_steps.append(
                    {
                        "controller": row["controller"],
                        "workflow_id": row["workflow_id"],
                        "overhead": overhead,
                        "migration_enabled": bool(step["workflow_state_migration_enabled"]),
                        "requested_mode": step["requested_migration_mode"],
                        "effective_mode": step["effective_migration_mode"],
                        "suppressed": bool(step["migration_suppressed"]),
                        "realized": bool(step["migration_prepare_realized"]),
                        "during_handoff": bool(step["migration_during_handoff"]),
                        "handoff_failed": bool(step["handoff_failed"]),
                        "cold_start": bool(step["cross_rsu_cold_start"]),
                        "state_migration_size_mb": float(
                            step.get("cache_event", {}).get("state_migration_size_mb", 0.0) or 0.0
                        ),
                    }
                )
    return {
        "sa_vs_mappo": {
            "pair_count": 36,
            "action_sequence_equal_count": sa_mappo_action_equal,
            "field_difference_counts": dict(sorted(sa_mappo_changes.items())),
            "all_differences_confined_to_seed_1403": all(
                int(item["seed"]) == 1403 for item in sa_mappo_examples
            ),
            "difference_rows": sa_mappo_examples,
        },
        "capacity_vs_shared": {
            "pair_count": len(capacity_pairs),
            "normalized_request_schedule_equal_count": schedules_equal,
            "exact_metric_and_action_row_count": exact_metric_action,
            "capacity_rejection_count_across_both_scenarios": capacity_rejections,
            "capacity_competition_eviction_count": capacity_evictions,
            "shared_adapter_reuse_prepare_eviction_count": shared_evictions,
            "capacity_competition_max_occupancy_rate": max(capacity_occupancy),
            "shared_adapter_reuse_prepare_max_occupancy_rate": max(shared_occupancy),
            "config_difference": "capacity_mb=280 versus 360; migration and repeated_reuse binding are identical",
            "scientific_interpretation": "paired capacity treatment whose observed contrast collapsed; not extra independent support",
            "pairs": capacity_pairs,
        },
        "low_reuse_no_migration": {
            "episode_count": len(low_reuse),
            "positive_migration_overhead_step_count": len(low_reuse_steps),
            "actual_state_migration_size_mb": sum(item["state_migration_size_mb"] for item in low_reuse_steps),
            "realized_migration_step_count": sum(
                item["realized"] or item["during_handoff"] for item in low_reuse_steps
            ),
            "suppressed_prepare_step_count": sum(item["suppressed"] for item in low_reuse_steps),
            "failed_handoff_cold_start_step_count": sum(
                item["handoff_failed"] and item["cold_start"] for item in low_reuse_steps
            ),
            "interpretation": "the nonzero field is the environment migration-overhead penalty for cross-RSU transfer without effective prepare/migrate, not realized workflow-state migration bytes",
        },
    }


def action_failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition_id"], row["seed"])].append(row)
    output = []
    for identity in sorted(groups, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
        group = groups[identity]
        actions = Counter()
        for row in group:
            actions.update(item for item in str(row["action_sequence"]).split("|") if item)
        output.append(
            {
                "condition_id": identity[0],
                "seed": identity[1],
                "episode_count": len(group),
                "completed_count": sum(bool(row["workflow_completed"]) for row in group),
                "request_failure_count": sum(int(row["request_failure_count"]) for row in group),
                "failure_categories": "|".join(
                    f"{name}:{count}"
                    for name, count in sorted(Counter(row["failure_category"] for row in group).items())
                ),
                "action_counts": "|".join(
                    f"{name}:{count}" for name, count in sorted(actions.items())
                ),
                "action_count": sum(actions.values()),
            }
        )
    return output


def pareto_audit(condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    learned = [
        row for row in condition_rows if row["condition_id"] != "crdcm_critical_path_heuristic"
    ]
    maximize = ("completion_rate", "continuity_episode_mean")
    minimize = (
        "handoff_failure_episode_mean",
        "transfer_mb_per_request_episode_mean",
        "migration_overhead_episode_mean",
    )
    domination: dict[str, list[str]] = defaultdict(list)
    for candidate in learned:
        for challenger in learned:
            if candidate is challenger:
                continue
            no_worse = all(challenger[field] >= candidate[field] for field in maximize) and all(
                challenger[field] <= candidate[field] for field in minimize
            )
            strict = any(challenger[field] > candidate[field] for field in maximize) or any(
                challenger[field] < candidate[field] for field in minimize
            )
            if no_worse and strict:
                domination[str(candidate["condition_id"])].append(str(challenger["condition_id"]))
    frontier = [
        str(row["condition_id"])
        for row in learned
        if not domination[str(row["condition_id"])]
    ]
    return {
        "metric_direction": {**{field: "maximize" for field in maximize}, **{field: "minimize" for field in minimize}},
        "learned_condition_frontier": frontier,
        "dominated_by": dict(sorted(domination.items())),
        "post_hoc_weighted_score_used": False,
        "heuristic_in_frontier": False,
        "heuristic_exclusion_reason": "one fixed heuristic has 12 episodes rather than 36 seed-replicated learned episodes and is not an independent 36-episode sample",
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    command_manifest_path = source_root / "command_manifest.json"
    command_manifest = load_json(command_manifest_path)
    completion = load_json(source_root / "completion_receipt.json")
    launch = load_json(source_root / "launch_receipt.json")
    state = load_json(source_root / "state.json")
    checkpoint_manifest_path = source_root / "checkpoint_manifest.json"
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    evaluation_receipt_path = source_root / "evaluation/evaluation_receipt.json"
    evaluation_receipt = load_json(evaluation_receipt_path)

    manifest_sha = sha256(command_manifest_path)
    identity_checks = {
        "completion_succeeded": completion["status"] == "SUCCEEDED" and completion["return_code"] == 0,
        "state_succeeded": state["status"] == "SUCCEEDED" and state["return_code"] == 0,
        "evaluation_succeeded": evaluation_receipt["status"] == "SUCCEEDED",
        "manifest_sha_bound": manifest_sha
        == completion["command_manifest_sha256"]
        == launch["command_manifest_sha256"],
        "child_count": completion["expected_child_count"] == completion["completed_child_count"] == 12,
        "all_child_return_codes_zero": all(item["return_code"] == 0 for item in completion["child_receipts"]),
        "automatic_retry_zero": completion["automatic_retry_count"] == command_manifest["automatic_retry_count"] == 0,
        "observed_data_not_formal_holdout": not command_manifest["formal"] and not command_manifest["holdout"],
        "algorithm_superiority_disallowed": not completion["algorithm_superiority_claim_allowed"],
        "evaluation_episode_count": int(evaluation_receipt["episode_count"]) == 156,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"source identity gate failed: {identity_checks}")

    integrity = verify_integrity(source_root)
    if integrity["status"] != "pass":
        raise RuntimeError("source artifact integrity verification failed")

    training_rows, training_summary, checkpoint_hashes = verify_training_and_checkpoints(
        source_root,
        command_manifest,
        checkpoint_manifest,
        evaluation_receipt,
    )
    if training_summary["status"] != "pass":
        raise RuntimeError("training/checkpoint audit failed")

    summaries: dict[tuple[str, int | None, str, str], dict[str, Any]] = {}
    episode_rows = []
    for path in sorted((source_root / "evaluation/episodes").glob("**/*.json")):
        summary = load_json(path)
        row = row_from_episode(path, summary)
        key = (
            str(row["condition_id"]),
            row["seed"],
            str(row["scenario_id"]),
            str(row["workflow_id"]),
        )
        if key in summaries:
            raise ValueError(f"duplicate evaluation identity: {key}")
        summaries[key] = summary
        episode_rows.append(row)
    if len(episode_rows) != 156:
        raise RuntimeError(f"unexpected raw episode count: {len(episode_rows)}")
    if not all(row["denominator_consistent"] for row in episode_rows):
        raise RuntimeError("raw request denominator mismatch")

    original_rows = compare_original_rows(source_root, episode_rows, checkpoint_hashes)
    original_aggregate = compare_original_aggregate(source_root, episode_rows)
    if original_rows["status"] != "pass" or original_aggregate["status"] != "pass":
        raise RuntimeError("independent recomputation differs from producer outputs")

    controller_seed_rows = grouped_rows(episode_rows, ("condition_id", "seed"))
    condition_rows = grouped_rows(episode_rows, ("condition_id",))
    scenario_rows = grouped_rows(episode_rows, ("condition_id", "scenario_id"))
    action_rows = action_failure_summary(episode_rows)
    trace_audit = pairwise_trace_audit(episode_rows, summaries)
    pareto = pareto_audit(condition_rows)

    expected_conditions = {
        "crdcm_full_sa": {1401, 1402, 1403},
        "crdcm_signal_off_sa": {1401, 1402, 1403},
        "crdcm_full_mappo": {1401, 1402, 1403},
        "crdcm_full_ppo": {1401, 1402, 1403},
    }
    pairing = {}
    for condition, seeds in expected_conditions.items():
        subset = [row for row in episode_rows if row["condition_id"] == condition]
        identities = {
            (int(row["seed"]), str(row["scenario_id"]), str(row["workflow_id"]))
            for row in subset
        }
        expected = {
            (seed, scenario, workflow)
            for seed in seeds
            for scenario in (
                "capacity_competition",
                "shared_adapter_reuse_prepare",
                "low_reuse_no_migration_negative",
                "ample_resource_negative",
            )
            for workflow in ("j_3", "j_8", "j_15")
        }
        pairing[condition] = {
            "observed_count": len(identities),
            "expected_count": len(expected),
            "missing": sorted(expected - identities),
            "extra": sorted(identities - expected),
            "status": "pass" if identities == expected else "fail",
        }
    exposure_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in episode_rows:
        exposure_groups[(row["scenario_id"], row["workflow_id"])].add(
            row["request_exposure_fingerprint"]
        )
    exposure_pairing = {
        f"{scenario}/{workflow}": {
            "unique_fingerprint_count": len(values),
            "fingerprints": sorted(values),
        }
        for (scenario, workflow), values in sorted(exposure_groups.items())
    }
    if any(item["unique_fingerprint_count"] != 1 for item in exposure_pairing.values()):
        raise RuntimeError("controllers did not receive identical paired exposures")

    budget = command_manifest["budget"]
    budget_audit = {
        "manifest_training_episode_cap": budget["training_episodes"],
        "observed_training_episodes": training_summary["training_episodes"],
        "manifest_training_step_cap": budget["training_environment_step_cap"],
        "observed_training_steps": training_summary["actual_training_steps"],
        "observed_training_updates": training_summary["training_updates"],
        "manifest_evaluation_episodes": budget["evaluation_episodes"],
        "observed_evaluation_episodes": len(episode_rows),
        "manifest_evaluation_step_cap": budget["evaluation_environment_step_cap"],
        "observed_evaluation_steps": sum(int(row["total_steps"]) for row in episode_rows),
        "manifest_full_episode_cap": budget["full_execution_episode_cap"],
        "observed_full_episode_count": training_summary["training_episodes"] + len(episode_rows),
        "manifest_full_step_cap": budget["full_execution_environment_step_cap"],
        "observed_full_steps": training_summary["actual_training_steps"]
        + sum(int(row["total_steps"]) for row in episode_rows),
    }
    budget_audit["within_all_caps"] = (
        budget_audit["observed_training_episodes"] <= budget_audit["manifest_training_episode_cap"]
        and budget_audit["observed_training_steps"] <= budget_audit["manifest_training_step_cap"]
        and budget_audit["observed_evaluation_episodes"] <= budget_audit["manifest_evaluation_episodes"]
        and budget_audit["observed_evaluation_steps"] <= budget_audit["manifest_evaluation_step_cap"]
        and budget_audit["observed_full_episode_count"] <= budget_audit["manifest_full_episode_cap"]
        and budget_audit["observed_full_steps"] <= budget_audit["manifest_full_step_cap"]
    )

    source_identity_checks = {}
    for role, entry in command_manifest["sources"].items():
        path = Path(str(entry["path"]))
        source_identity_checks[role] = {
            "path": str(path),
            "expected_size_bytes": int(entry["size_bytes"]),
            "observed_size_bytes": path.stat().st_size,
            "expected_sha256": str(entry["sha256"]),
            "observed_sha256": sha256(path),
        }
        source_identity_checks[role]["status"] = (
            source_identity_checks[role]["expected_size_bytes"]
            == source_identity_checks[role]["observed_size_bytes"]
            and source_identity_checks[role]["expected_sha256"]
            == source_identity_checks[role]["observed_sha256"]
        )
    bound_file_checks = {}
    for name, entry in (
        ("config", {"path": command_manifest["config_path"], "sha256": command_manifest["config_sha256"]}),
        ("runtime_config", command_manifest["runtime_config"]),
        ("window_plan", command_manifest["window_plan"]),
    ):
        path = Path(str(entry["path"]))
        observed = sha256(path)
        bound_file_checks[name] = {
            "path": str(path),
            "expected_sha256": str(entry["sha256"]),
            "observed_sha256": observed,
            "status": observed == str(entry["sha256"]),
        }
    stderr_paths = [source_root / "outer.stderr.log", source_root / "logs/evaluation.stderr.log"]
    stderr_paths.extend(sorted((source_root / "logs").glob("crdcm_*.stderr.log")))
    log_audit = {
        "stderr_file_count": len(stderr_paths),
        "empty_stderr_file_count": sum(path.stat().st_size == 0 for path in stderr_paths),
        "nonempty_stderr_paths": [str(path) for path in stderr_paths if path.stat().st_size != 0],
    }
    log_audit["status"] = log_audit["stderr_file_count"] == 14 and not log_audit["nonempty_stderr_paths"]

    learned_summaries = [
        summary
        for (condition, _, _, _), summary in summaries.items()
        if condition != "crdcm_critical_path_heuristic"
    ]
    embedded_checkpoint_count = 0
    for summary in learned_summaries:
        embedded = summary["run_info"].get("checkpoint_metadata", {})
        if (
            embedded.get("checkpoint_path")
            and int(embedded.get("episodes", 0)) == 64
            and int(embedded.get("update_count", 0)) == 16
        ):
            embedded_checkpoint_count += 1
    trace_checkpoint_reconciliation = {
        "learned_raw_episode_count": len(learned_summaries),
        "raw_episode_self_contained_checkpoint_identity_count": embedded_checkpoint_count,
        "indirectly_reconciled_count": len(learned_summaries),
        "indirect_reconciliation_chain": "raw condition_id+seed -> paired CSV checkpoint_sha256 -> checkpoint manifest latest.pt -> evaluation before/after SHA-256",
        "limitation": "raw episode run_info.checkpoint_metadata is the in-memory override placeholder (empty path, episodes=0, update_count=0), so individual raw JSON files are not self-contained checkpoint provenance envelopes",
    }
    denominator_audit = {
        "episode_count": len(episode_rows),
        "consistent_episode_count": sum(bool(row["denominator_consistent"]) for row in episode_rows),
        "total_external_requests": sum(int(row["request_count"]) for row in episode_rows),
        "completed_episode_count": sum(bool(row["workflow_completed"]) for row in episode_rows),
        "delay_available_count": sum(bool(row["delay_available"]) for row in episode_rows),
    }
    denominator_audit["status"] = (
        denominator_audit["episode_count"] == denominator_audit["consistent_episode_count"] == 156
        and denominator_audit["completed_episode_count"] == denominator_audit["delay_available_count"]
    )

    protocol_audit = {
        "study_version": command_manifest["study_version"],
        "execution_scope": command_manifest["execution_scope"],
        "condition_count": len(command_manifest["conditions"]),
        "seed_count": len(command_manifest["seeds"]),
        "command_count": len(command_manifest["commands"]),
        "unique_command_identity_count": len(
            {
                (row["condition_id"], int(row["seed"]))
                for row in command_manifest["commands"]
            }
        ),
        "fixed_endpoint": checkpoint_manifest["fixed_endpoint"],
        "metric_based_selection": checkpoint_manifest["metric_based_selection"],
        "source_data_identity": source_identity_checks,
        "bound_file_identity": bound_file_checks,
        "log_audit": log_audit,
    }
    protocol_audit["status"] = (
        protocol_audit["condition_count"] == 4
        and protocol_audit["seed_count"] == 3
        and protocol_audit["command_count"] == protocol_audit["unique_command_identity_count"] == 12
        and protocol_audit["fixed_endpoint"] == "latest.pt after episode 64 / update 16"
        and not protocol_audit["metric_based_selection"]
        and all(item["status"] for item in source_identity_checks.values())
        and all(item["status"] for item in bound_file_checks.values())
        and log_audit["status"]
    )

    audit = {
        "audit_version": AUDIT_VERSION,
        "reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "literature_cutoff": LITERATURE_CUTOFF,
        "target_venue": TARGET_VENUE,
        "artifact_run_id": source_root.name,
        "policy_version": POLICY_VERSION,
        "source_execution_git_commit": command_manifest["git_commit"],
        "source_execution_git_tree": command_manifest["git_tree"],
        "review_git_commit_before_artifact_commit": git_value("rev-parse", "HEAD"),
        "auditor_identity": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "python_executable": os.path.realpath(os.sys.executable),
            "torch_version": torch.__version__,
        },
        "evidence_level": "E3_REPRODUCED_OBSERVED_DATA_DEVELOPMENT_PILOT_NOT_HOLDOUT",
        "verdict": "Not TMC-ready",
        "source_identity": {
            "source_root": str(source_root),
            "command_manifest_path": str(command_manifest_path),
            "command_manifest_sha256": manifest_sha,
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
            "evaluation_receipt_path": str(evaluation_receipt_path),
            "evaluation_receipt_sha256": sha256(evaluation_receipt_path),
            "checks": identity_checks,
        },
        "integrity": integrity,
        "protocol_audit": protocol_audit,
        "budget": budget_audit,
        "training_checkpoint_summary": training_summary,
        "trace_checkpoint_reconciliation": trace_checkpoint_reconciliation,
        "denominator_and_delay_audit": denominator_audit,
        "producer_csv_reconciliation": original_rows,
        "producer_aggregate_reconciliation": original_aggregate,
        "pairing": pairing,
        "exposure_pairing": exposure_pairing,
        "raw_window_independence": {
            "outer_independent_window_count": 3,
            "manifest_interval_audit": command_manifest["window_interval_audit"],
            "seed_role": "within-window repeated training/evaluation, not an independent raw mobility cluster",
            "scenario_role": "paired treatment on the same three raw windows, not an independent raw mobility cluster",
            "workflow_role": "one workflow is bound to each raw window and reused by all controllers",
            "statistical_boundary": "no row-level or seed-level independence claim; three outer windows imply very low power",
        },
        "pairwise_trace_audit": trace_audit,
        "pareto": pareto,
        "claim_boundary": command_manifest["claim_boundary"],
        "safe_claims": [
            "The frozen observed-data development matrix completed and its 156 raw evaluation episodes reproduce the producer CSV and aggregate without mismatch.",
            "On the fixed episode-mean summary, full PPO completed 27/36 episodes versus 18/36 for full SA, while full SA used less transfer per request (24.063036 versus 31.613072 MB); this is a descriptive Pareto trade-off, not superiority.",
            "The added CRDCM signal changed SA outcomes, but the direction was seed-dependent: full SA completed 18/36 versus 14/36 for signal-off while its continuity mean was lower (0.764343 versus 0.846006).",
            "Full SA weakly dominates full MAPPO on the recorded learned-condition episode-mean vector because completion/cost metrics match while SA continuity is higher; this remains an observed-data development result.",
        ],
        "prohibited_claims": [
            "CRDCM, SA-GHMAPPO, MAPPO, or PPO has formal or independently generalizable superiority.",
            "The signal-on versus signal-off contrast establishes a stable causal benefit.",
            "The capacity_competition scenario provides independent capacity-stress evidence distinct from shared_adapter_reuse_prepare.",
            "Seed 1403 failures prove insufficient training, convergence failure, or an architecture defect.",
            "The 12 heuristic episodes are 36 independent comparisons or are directly interchangeable with the three-seed learned totals.",
            "The no-migration scenario has zero migration-overhead penalty or that its nonzero penalty represents realized state-migration bytes.",
        ],
        "hard_blockers": [
            "No formal or hidden holdout evaluation exists; all evaluation is observed-data development resubstitution.",
            "Only three raw mobility windows are outer independent units; seeds, scenarios, and workflows are not additional independent clusters.",
            "The intended capacity contrast collapsed: capacity_competition and shared_adapter_reuse_prepare have identical normalized request schedules and identical metric/action rows for all 39 paired controllers, with zero capacity rejection.",
            "Strong seed interaction remains: full SA and full MAPPO are 9/12, 9/12, 0/12 across seeds, while signal-off SA is 4/12, 1/12, 9/12.",
        ],
        "major_concerns": [
            "All 144 learned raw episode JSON files require indirect checkpoint linkage through condition/seed, the paired CSV, and checkpoint manifest because their embedded checkpoint metadata is an in-memory override placeholder.",
            "The strong heuristic has only 12 fixed episodes; it cannot be presented as 36 independent observations or pooled as though it had three seed replicates.",
            "Delay is defined only for completed workflows: 83/156 episodes have delay values, so delay comparisons must report coverage and remain conditional on completion.",
            "Latest checkpoints and update_0016 checkpoints are not byte-identical separate serializations, although all 12 loaded payloads are recursively equal and carry update_count=16; citations must use the exact manifest SHA of latest.pt.",
        ],
        "minor_concerns": [
            "The scenario name low_reuse_no_migration_negative is easy to misread: migration is disabled, but the environment still records a nonzero migration-overhead reward penalty for failed cross-RSU continuity without effective preparation.",
            "Episode-mean and request-weighted continuity/transfer differ because workflows have unequal request counts; tables must label the aggregation unit explicitly.",
        ],
        "scorecard": {
            "Novelty and nearest literature": "N/S — this audit does not establish novelty and CRDCM novelty remains UNVERIFIED",
            "Technical correctness and modeling": "N/S — trace semantics were audited, but this is not a complete method/model review",
            "Baseline fairness and independence": "N/S — only observed-data resubstitution and three outer windows are available",
            "Experimental design, statistics, holdout": "N/S — no formal/holdout stage and no defensible independent-cluster inference",
            "Mechanism realization": "N/S — several mechanism traces exist, but the capacity contrast did not realize an outcome difference",
            "Robustness, generalization, scalability": "N/S — not present in this matrix",
            "Reproducibility and claim completeness": "N/S for TMC score — the matrix itself reached E3 independent re-aggregation, but paper-level evidence is incomplete",
        },
        "evidence_inventory": {
            "source_artifact_file_count_excluding_integrity_manifest": integrity["observed_file_count_excluding_manifest"],
            "raw_training_episode_count": training_summary["training_episodes"],
            "raw_evaluation_episode_count": len(episode_rows),
            "checkpoint_count": training_summary["checkpoint_identity_count"],
            "paired_csv_recomputed_without_mismatch": original_rows["mismatch_count"] == 0,
            "aggregate_recomputed_without_mismatch": original_aggregate["mismatch_count"] == 0,
        },
        "first_order_next_step": "Before any new algorithm tuning, redesign and freeze one outcome-blind capacity-stress evaluation whose trace actually activates capacity rejection or eviction pressure, then evaluate the already frozen checkpoints on additional nonoverlapping outer windows; do not reopen the present results for seed or checkpoint selection.",
    }

    write_csv(output / "independent_episode_rows.csv", episode_rows)
    write_csv(output / "condition_aggregate.csv", condition_rows)
    write_csv(output / "condition_seed_aggregate.csv", controller_seed_rows)
    write_csv(output / "condition_scenario_aggregate.csv", scenario_rows)
    write_csv(output / "action_failure_summary.csv", action_rows)
    write_csv(output / "training_checkpoint_audit.csv", training_rows)
    write_json(output / "pairwise_trace_audit.json", trace_audit)
    write_json(output / "independent_review.json", audit)

    target = output / "artifact_integrity_manifest.json"
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != target:
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        target,
        {
            "artifact_integrity_manifest_version": "1.0.0",
            "self_excluding": True,
            "status": "pass",
            "file_count": len(files),
            "files": files,
        },
    )
    print(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "output_dir": str(output),
                "raw_episode_count": len(episode_rows),
                "producer_csv_mismatch_count": original_rows["mismatch_count"],
                "producer_aggregate_mismatch_count": original_aggregate["mismatch_count"],
                "training_episodes": training_summary["training_episodes"],
                "training_steps": training_summary["actual_training_steps"],
                "training_updates": training_summary["training_updates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
