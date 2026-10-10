"""Analyze the single-variable SA service-feasible event-target experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json  # noqa: E402
from scripts.run_calibrated_workflow_service_feasible_event_target_ab import (  # noqa: E402
    CANDIDATE_ARM,
    CONTROL_ARM,
    _load_json,
    _read_csv,
)


PAIR_KEYS = ("method", "seed", "split", "design_id", "window_id", "workflow_id", "source_segment_id")
METRICS = (
    "workflow_completion_rate", "on_time_workflow_completion_rate",
    "service_failure_rate", "service_failures", "recompute_seconds",
    "model_prepare_mb", "state_transfer_mb", "input_transfer_mb",
    "total_transfer_mb", "reward",
)


def _float(row: dict[str, Any], field: str) -> float:
    value = row.get(field, "")
    if value in {"", None}:
        raise RuntimeError(f"missing numeric field: {field}")
    return float(value)


def _validate_source(source_root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    receipt = _load_json(source_root / "completion_receipt.json")
    manifest = _load_json(source_root / "run_manifest.json")
    if (
        receipt.get("status") != "complete"
        or int(receipt.get("learned_cells", -1)) != 5
        or int(receipt.get("learned_environment_steps", -1)) != 28800
        or int(receipt.get("learned_optimizer_steps", -1)) != 3840
        or int(receipt.get("new_evaluation_episodes", -1)) != 200
    ):
        raise RuntimeError("candidate completion identity mismatch")
    integrity = _load_json(source_root / "artifact_integrity.json")
    for item in integrity["files"]:
        path = source_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"candidate integrity mismatch: {item['path']}")
    rows = _read_csv(source_root / "evaluation_rows.csv")
    if len(rows) != 1040:
        raise RuntimeError("candidate combined evaluation row count mismatch")
    return manifest, rows


def _pair(
    rows: list[dict[str, str]],
    *,
    control_arm: str = CONTROL_ARM,
    candidate_arm: str = CANDIDATE_ARM,
) -> list[dict[str, Any]]:
    result = []
    for view in ("selected", "update96"):
        old = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in rows
            if row["event_target_arm"] == control_arm
            and row["checkpoint_view"] == view and row["method"] == "sa_ghmappo"
        }
        new = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in rows
            if row["event_target_arm"] == candidate_arm
            and row["checkpoint_view"] == view and row["method"] == "sa_ghmappo"
        }
        if old.keys() != new.keys() or len(old) != 100:
            raise RuntimeError(f"candidate pairing mismatch: {view}")
        for key in sorted(old):
            before, after = old[key], new[key]
            item: dict[str, Any] = {field: value for field, value in zip(PAIR_KEYS, key)}
            item["checkpoint_view"] = view
            for metric in METRICS:
                item[f"control_{metric}"] = _float(before, metric)
                item[f"candidate_{metric}"] = _float(after, metric)
                item[f"delta_{metric}"] = item[f"candidate_{metric}"] - item[f"control_{metric}"]
            common = item["control_workflow_completion_rate"] == item["candidate_workflow_completion_rate"] == 1.0
            item["common_completed"] = common
            item["delta_common_completed_elapsed_seconds"] = (
                _float(after, "completed_sample_elapsed_seconds")
                - _float(before, "completed_sample_elapsed_seconds")
                if common else ""
            )
            result.append(item)
    return result


def _paired_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[(row["checkpoint_view"], row["split"])].append(row)
        groups[(row["checkpoint_view"], "combined")].append(row)
    result = []
    for (view, split), group in sorted(groups.items()):
        common = [row for row in group if row["common_completed"]]
        item: dict[str, Any] = {
            "checkpoint_view": view, "split": split, "pair_n": len(group),
            "common_completed_n": len(common),
        }
        for metric in METRICS:
            item[f"control_mean_{metric}"] = float(np.mean([row[f"control_{metric}"] for row in group]))
            item[f"candidate_mean_{metric}"] = float(np.mean([row[f"candidate_{metric}"] for row in group]))
            item[f"mean_delta_{metric}"] = float(np.mean([row[f"delta_{metric}"] for row in group]))
        item["mean_delta_common_completed_elapsed_seconds"] = (
            float(np.mean([float(row["delta_common_completed_elapsed_seconds"]) for row in common]))
            if common else ""
        )
        result.append(item)
    return result


def _seed_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[(row["checkpoint_view"], row["seed"], row["split"])].append(row)
        groups[(row["checkpoint_view"], row["seed"], "combined")].append(row)
    result = []
    for (view, seed, split), group in sorted(groups.items()):
        item: dict[str, Any] = {"checkpoint_view": view, "seed": seed, "split": split, "episode_n": len(group)}
        for metric in METRICS:
            item[f"mean_delta_{metric}"] = float(np.mean([row[f"delta_{metric}"] for row in group]))
        result.append(item)
    return result


def _behavior_summary(rows: list[dict[str, str]], arm: str, view: str) -> dict[str, Any]:
    by_episode: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] != "sa_ghmappo":
            continue
        by_episode[(row["split"], row["method"], row["seed"], row["design_id"])].append(row)
    current_missing = current_missing_action4 = state_commits = 0
    reuse = recompute = stale = current_fail = fallback = 0
    for episode_rows in by_episode.values():
        ordered = sorted(episode_rows, key=lambda row: int(row["step_index"]))
        previous_rsu = None
        for row in ordered:
            action = int(row["executed_action"])
            if row["current_bundle_ready"] == "False":
                current_missing += 1
                if action == 4:
                    current_missing_action4 += 1
            if action == 4 and row["migration_success"] == "True":
                state_commits += 1
            current_rsu = row["current_rsu_id"]
            if previous_rsu is not None and current_rsu != previous_rsu:
                if action == 2:
                    fallback += 1
                elif row["state_ready"] == "True":
                    reuse += 1
                elif row["service_completed"] != "True":
                    current_fail += 1
                else:
                    recompute += 1
                    if row.get("current_prepared_exists") == "True" and row.get("current_prepared_valid") != "True":
                        stale += 1
            previous_rsu = current_rsu
    return {
        "event_target_arm": arm, "checkpoint_view": view,
        "episode_n": len(by_episode),
        "behavior_step_n": sum(len(episode) for episode in by_episode.values()),
        "current_missing_decisions": current_missing,
        "current_missing_action4": current_missing_action4,
        "current_missing_action4_rate": (
            float(current_missing_action4) / current_missing if current_missing else 0.0
        ),
        "state_commits": state_commits, "handoff_prepared_reuse": reuse,
        "handoff_recompute": recompute, "handoff_prepared_stale": stale,
        "handoff_current_service_failed": current_fail, "handoff_vehicle_fallback": fallback,
    }


def _verdict(summary: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [row for row in summary if row["split"] in {"regression", "frozen_check"}]
    required_keys = {(row["checkpoint_view"], row["split"]) for row in required}
    expected_keys = {
        (view, split)
        for view in ("selected", "update96")
        for split in ("regression", "frozen_check")
    }
    if required_keys != expected_keys or len(required) != len(expected_keys):
        raise RuntimeError("verdict requires both checkpoint views and both frozen development splits")
    completion_decrease = any(row["mean_delta_workflow_completion_rate"] < 0.0 for row in required)
    service_boundary_ok = all(
        row["mean_delta_on_time_workflow_completion_rate"] >= 0.0
        and row["mean_delta_service_failure_rate"] <= 0.0
        for row in required
    )
    strict_each_view = all(
        any(
            row["checkpoint_view"] == view
            and row["split"] == "combined"
            and (
                row["mean_delta_on_time_workflow_completion_rate"] > 0.0
                or row["mean_delta_service_failure_rate"] < 0.0
            )
            for row in summary
        )
        for view in ("selected", "update96")
    )
    seed_direction = {}
    for view in ("selected", "update96"):
        combined = [row for row in seed_rows if row["checkpoint_view"] == view and row["split"] == "combined"]
        if len(combined) != 5 or len({str(row["seed"]) for row in combined}) != 5:
            raise RuntimeError(f"verdict requires five distinct seeds for {view}")
        improved = sum(
            row["mean_delta_on_time_workflow_completion_rate"] >= 0.0
            and row["mean_delta_service_failure_rate"] <= 0.0
            and (
                row["mean_delta_on_time_workflow_completion_rate"] > 0.0
                or row["mean_delta_service_failure_rate"] < 0.0
            )
            for row in combined
        )
        worsened = sum(
            row["mean_delta_on_time_workflow_completion_rate"] < 0.0
            or row["mean_delta_service_failure_rate"] > 0.0
            for row in combined
        )
        seed_direction[view] = {
            "improved": improved,
            "worsened_or_tradeoff": worsened,
            "unchanged": len(combined) - improved - worsened,
        }
    seed_gate = all(
        value["improved"] >= 3 and value["worsened_or_tradeoff"] <= 1
        for value in seed_direction.values()
    )
    if completion_decrease:
        status = "FAIL"
    elif service_boundary_ok and strict_each_view and seed_gate:
        status = "PASS"
    else:
        status = "MIXED"
    return {
        "status": status, "completion_decrease": completion_decrease,
        "on_time_and_failure_boundary_ok_in_each_view_and_split": service_boundary_ok,
        "strict_on_time_or_failure_improvement_in_each_view": strict_each_view,
        "seed_direction": seed_direction, "seed_gate_passed": seed_gate,
        "no_second_candidate_or_budget_extension": True,
    }


def analyze(
    source_root: Path,
    control_root: Path,
    analysis_root: Path,
    *,
    control_arm: str = CONTROL_ARM,
    candidate_arm: str = CANDIDATE_ARM,
    analysis_schema: str = "calibrated_workflow_service_feasible_event_target_ab_analysis_v1",
) -> None:
    manifest, rows = _validate_source(source_root)
    if analysis_root.exists():
        raise FileExistsError(f"create-only analysis root exists: {analysis_root}")
    analysis_root.mkdir(parents=True)
    pairs = _pair(
        rows,
        control_arm=control_arm,
        candidate_arm=candidate_arm,
    )
    summary = _paired_summary(pairs)
    seed_rows = _seed_summary(pairs)
    control_selected = _read_csv(control_root / "new_selected_behavior_ledger.csv")
    control_fixed = _read_csv(control_root / "new_update96_behavior_ledger.csv")
    candidate_selected = _read_csv(source_root / "candidate_selected_behavior_ledger.csv")
    candidate_fixed = _read_csv(source_root / "candidate_update96_behavior_ledger.csv")
    behavior = [
        _behavior_summary(control_selected, control_arm, "selected"),
        _behavior_summary(control_fixed, control_arm, "update96"),
        _behavior_summary(candidate_selected, candidate_arm, "selected"),
        _behavior_summary(candidate_fixed, candidate_arm, "update96"),
    ]
    verdict = _verdict(summary, seed_rows)
    _write_csv(analysis_root / "paired_episode_rows.csv", pairs)
    _write_csv(analysis_root / "paired_summary.csv", summary)
    _write_csv(analysis_root / "seed_split_summary.csv", seed_rows)
    _write_csv(analysis_root / "behavior_summary.csv", behavior)
    _write_csv(
        analysis_root / "strong_baseline_reference.csv",
        [row for row in rows if row["event_target_arm"] in {control_arm, "historical_rule_reference"} and row["method"] != "sa_ghmappo"],
    )
    _write_json(analysis_root / "gate_verdict.json", verdict)
    _write_json(analysis_root / "analysis_manifest.json", {
        "schema_version": analysis_schema,
        "source_run_id": source_root.name,
        "source_run_manifest_sha256": _sha256(source_root / "run_manifest.json"),
        "source_artifact_integrity_sha256": _sha256(source_root / "artifact_integrity.json"),
        "scientific_commit": manifest["git_commit"], "paired_rows": len(pairs),
        "primary_view": "selected", "secondary_view": "update96",
        "control_reused_without_reevaluation": True,
        "claim_boundary": manifest["claim_boundary"],
    })
    _integrity(analysis_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    args = parser.parse_args()
    analyze(Path(args.source_root).resolve(), Path(args.control_root).resolve(), Path(args.analysis_root).resolve())


if __name__ == "__main__":
    main()
