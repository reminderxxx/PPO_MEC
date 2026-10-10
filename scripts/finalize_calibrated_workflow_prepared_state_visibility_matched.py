"""Create a fail-closed v2 post-analysis for the completed matched run."""

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

from scripts.analyze_calibrated_workflow_prepared_state_visibility_matched import (  # noqa: E402
    PAIR_KEYS,
    _validate_source,
)
from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json  # noqa: E402


LEARNED = {"sa_ghmappo", "mappo", "ppo", "dt_handoff_drl"}
ARMS = ("prepared_state_hidden_v3", "prepared_state_visible_v4")
VIEWS = ("selected", "update96")
METRICS = (
    "workflow_completion_rate",
    "on_time_workflow_completion_rate",
    "service_failure_rate",
    "service_failures",
    "recompute_seconds",
    "model_prepare_mb",
    "state_transfer_mb",
    "input_transfer_mb",
    "total_transfer_mb",
    "reward",
)


def _float(row: dict[str, Any], field: str) -> float:
    value = row.get(field, "")
    if value in {"", None}:
        raise RuntimeError(f"missing numeric field: {field}")
    return float(value)


def _group_summary(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        item: dict[str, Any] = {field: value for field, value in zip(fields, key)}
        item.update({
            "episode_n": len(group),
            "completion_n": sum(_float(row, "workflow_completion_rate") == 1.0 for row in group),
            "on_time_n": sum(_float(row, "on_time_workflow_completion_rate") == 1.0 for row in group),
            "failure_episode_n": sum(_float(row, "service_failure_rate") == 1.0 for row in group),
            "service_failure_count": int(sum(_float(row, "service_failures") for row in group)),
            "source_window_n": len({row["window_id"] for row in group}),
            "seed_n": len({row["seed"] for row in group}),
        })
        for metric in METRICS:
            item[f"mean_{metric}"] = float(np.mean([_float(row, metric) for row in group]))
        completed = [
            _float(row, "completed_sample_elapsed_seconds") for row in group
            if _float(row, "workflow_completion_rate") == 1.0
        ]
        item["completed_elapsed_n"] = len(completed)
        item["mean_completed_elapsed_seconds"] = float(np.mean(completed)) if completed else ""
        result.append(item)
    return result


def _pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    learned = [row for row in rows if row["method"] in LEARNED]
    result = []
    for view in VIEWS:
        old = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in learned
            if row["checkpoint_view"] == view and row["observation_arm"] == ARMS[0]
        }
        new = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in learned
            if row["checkpoint_view"] == view and row["observation_arm"] == ARMS[1]
        }
        if old.keys() != new.keys() or len(old) != 400:
            raise RuntimeError(f"pairing coverage mismatch: {view}")
        for key in sorted(old):
            before, after = old[key], new[key]
            item: dict[str, Any] = {field: value for field, value in zip(PAIR_KEYS, key)}
            item["checkpoint_view"] = view
            for metric in METRICS:
                item[f"old_{metric}"] = _float(before, metric)
                item[f"new_{metric}"] = _float(after, metric)
                item[f"delta_{metric}"] = item[f"new_{metric}"] - item[f"old_{metric}"]
            old_complete = item["old_workflow_completion_rate"] == 1.0
            new_complete = item["new_workflow_completion_rate"] == 1.0
            item["common_completed"] = old_complete and new_complete
            item["old_only_completed"] = old_complete and not new_complete
            item["new_only_completed"] = new_complete and not old_complete
            item["delta_common_completed_elapsed_seconds"] = (
                _float(after, "completed_sample_elapsed_seconds")
                - _float(before, "completed_sample_elapsed_seconds")
                if item["common_completed"] else ""
            )
            result.append(item)
    return result


def _paired_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["checkpoint_view"], row["method"], row["split"])].append(row)
        grouped[(row["checkpoint_view"], row["method"], "combined")].append(row)
    result = []
    for (view, method, split), group in sorted(grouped.items()):
        common = [row for row in group if row["common_completed"]]
        item: dict[str, Any] = {
            "checkpoint_view": view, "method": method, "split": split,
            "pair_n": len(group), "common_completed_n": len(common),
            "old_only_completed_n": sum(row["old_only_completed"] for row in group),
            "new_only_completed_n": sum(row["new_only_completed"] for row in group),
        }
        for metric in METRICS:
            item[f"old_mean_{metric}"] = float(np.mean([row[f"old_{metric}"] for row in group]))
            item[f"new_mean_{metric}"] = float(np.mean([row[f"new_{metric}"] for row in group]))
            item[f"mean_delta_{metric}"] = float(np.mean([row[f"delta_{metric}"] for row in group]))
        item["mean_delta_common_completed_elapsed_seconds"] = (
            float(np.mean([float(row["delta_common_completed_elapsed_seconds"]) for row in common]))
            if common else ""
        )
        result.append(item)
    return result


def _gate_verdict(summary: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {
        (row["checkpoint_view"], row["method"], row["split"]): row
        for row in summary
    }
    selected = lookup[("selected", "sa_ghmappo", "frozen_check")]
    fixed = lookup[("update96", "sa_ghmappo", "frozen_check")]
    selected_failure_worse = selected["mean_delta_service_failure_rate"] > 0.0
    on_time_not_both = not (
        selected["mean_delta_on_time_workflow_completion_rate"] > 0.0
        and fixed["mean_delta_on_time_workflow_completion_rate"] > 0.0
    )
    any_method_failure_worse = any(
        row["split"] == "frozen_check" and row["mean_delta_service_failure_rate"] > 0.0
        for row in summary
    )
    rejected = selected_failure_worse or on_time_not_both or any_method_failure_worse
    return {
        "interface_correctness": "accepted_as_shared_observation_contract_fix",
        "performance_candidate": "rejected_by_preregistered_dual_view_gate" if rejected else "accepted",
        "reasons": {
            "sa_selected_frozen_failure_worsened": selected_failure_worse,
            "sa_on_time_improvement_not_present_in_both_views": on_time_not_both,
            "any_learned_method_frozen_failure_worsened": any_method_failure_worse,
        },
        "failure_definition": {
            "service_failure_rate": "episode indicator: 1 iff summary.service_failures > 0",
            "service_failures": "count of failed service attempts within the episode",
            "not_equal_to": "1 - workflow_completion_rate",
        },
        "no_threshold_change_after_results": True,
        "new_training_or_evaluation": 0,
    }


def finalize(source_root: Path, analysis_root: Path) -> None:
    manifest, rows = _validate_source(source_root)
    if analysis_root.exists():
        raise FileExistsError(f"create-only analysis root exists: {analysis_root}")
    learned = [row for row in rows if row["method"] in LEARNED]
    rules = [row for row in rows if row["method"] not in LEARNED]
    if len(learned) != 1600 or len(rules) != 40:
        raise RuntimeError("learned/rule coverage mismatch")
    analysis_root.mkdir(parents=True)
    arm_split = _group_summary(learned, ("observation_arm", "checkpoint_view", "method", "split"))
    seed_split = _group_summary(learned, ("observation_arm", "checkpoint_view", "method", "seed", "split"))
    rule_summary = _group_summary(rules, ("method", "split"))
    pairs = _pair_rows(rows)
    paired_summary = _paired_summary(pairs)
    verdict = _gate_verdict(paired_summary)
    _write_csv(analysis_root / "arm_split_summary.csv", arm_split)
    _write_csv(analysis_root / "seed_split_summary.csv", seed_split)
    _write_csv(analysis_root / "rule_reference_summary.csv", rule_summary)
    _write_csv(analysis_root / "paired_episode_rows.csv", pairs)
    _write_csv(analysis_root / "paired_summary.csv", paired_summary)
    _write_json(analysis_root / "gate_verdict.json", verdict)
    _write_json(analysis_root / "analysis_manifest.json", {
        "schema_version": "calibrated_workflow_prepared_state_visibility_matched_analysis_v2",
        "source_run_id": source_root.name,
        "source_run_manifest_sha256": _sha256(source_root / "run_manifest.json"),
        "source_artifact_integrity_sha256": _sha256(source_root / "artifact_integrity.json"),
        "scientific_commit": manifest["git_commit"],
        "source_rows": len(rows), "learned_rows": len(learned), "rule_rows": len(rules),
        "paired_rows": len(pairs), "pair_keys": list(PAIR_KEYS),
        "primary_view": "selected", "secondary_view": "update96",
        "postprocess_only": True, "new_training_or_evaluation": 0,
        "claim_boundary": manifest["claim_boundary"],
    })
    _integrity(analysis_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    args = parser.parse_args()
    finalize(Path(args.source_root).resolve(), Path(args.analysis_root).resolve())


if __name__ == "__main__":
    main()
