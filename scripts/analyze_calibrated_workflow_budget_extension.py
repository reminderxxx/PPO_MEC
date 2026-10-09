"""Create a read-only, Python-3.9-compatible budget-extension analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json  # noqa: E402
from scripts.run_calibrated_workflow_strong_baselines import LEARNED_METHODS, _load_json, _read_csv  # noqa: E402


DEFAULT_CONFIG = "configs/experiment/calibrated_workflow_strong_baselines_budget_extension_analysis_v1.json"
PAIR_KEYS = (
    "split", "method", "seed", "design_id", "window_id", "workflow_id",
    "source_segment_id", "source_frame_offset", "source_window_length",
    "source_time_start", "source_time_end",
)
SIGNAL_KEYS = ("method", "seed", "global_cell_step")
TRAJECTORY_FIELDS = (
    "training_episode_index", "rollout_segment_index", "action", "reward",
    "terminated", "truncated", "current_bundle_ready", "prediction_provenance",
)
PAIR_METRICS = (
    "on_time_workflow_completion_rate", "workflow_completion_rate",
    "unfinished_after_deadline_rate", "service_failure_rate", "service_failures",
    "failed_service_attempt_seconds_proxy", "node_coverage_rate",
    "modeled_completion_seconds",
    "total_transfer_mb", "model_prepare_mb", "state_transfer_mb", "input_transfer_mb",
    "recompute_seconds", "invalid_prepare_attempts", "max_consecutive_no_progress_steps",
    "action_0", "action_1", "action_2", "action_3", "action_4", "reward",
)
OPTIMIZER_METRICS = (
    "actor_loss", "env_action_ppo_loss", "value_loss", "auxiliary_loss", "total_loss",
    "entropy", "approx_kl", "clip_fraction", "policy_grad_norm",
    "weighted_value_grad_norm", "weighted_auxiliary_grad_norm",
    "pre_clip_total_grad_norm", "post_clip_total_grad_norm_upper_bound", "global_clip_scale",
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def _key(row: dict[str, Any], fields: Sequence[str], label: str) -> tuple[str, ...]:
    missing = [field for field in fields if field not in row or str(row[field]) == ""]
    if missing:
        raise RuntimeError(f"{label} missing pairing key(s): {','.join(missing)}")
    return tuple(str(row[field]) for field in fields)


def _unique_index(
    rows: Iterable[dict[str, Any]], fields: Sequence[str], label: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = _key(row, fields, label)
        if key in indexed:
            raise RuntimeError(f"{label} duplicate pairing key: {key}")
        indexed[key] = row
    return indexed


def _pair_rows(
    short_rows: Iterable[dict[str, Any]],
    long_rows: Iterable[dict[str, Any]],
    fields: Sequence[str] = PAIR_KEYS,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair by explicit identity, rejecting missing, duplicate, or unequal key sets."""
    short = _unique_index(short_rows, fields, "short")
    long = _unique_index(long_rows, fields, "long")
    if set(short) != set(long):
        short_only = sorted(set(short) - set(long))[:3]
        long_only = sorted(set(long) - set(short))[:3]
        raise RuntimeError(
            f"paired key-set mismatch: short_only={short_only}, long_only={long_only}"
        )
    return [(short[key], long[key]) for key in sorted(short)]


def _group(
    rows: Iterable[dict[str, Any]], fields: Sequence[str],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row, fields, "group")].append(row)
    return grouped


def _canonical_rows_sha256(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    ordered = sorted(rows, key=lambda row: _key(row, fields, "canonical"))
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_inventory(root: Path) -> int:
    inventory = _load_json(root / "artifact_integrity.json")
    for item in inventory["files"]:
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"scientific artifact integrity mismatch: {item['path']}")
    return len(inventory["files"])


def _validate_frozen_sources(
    source_root: Path, short_root: Path, design: dict[str, Any], expected_analysis_commit: str,
) -> dict[str, Any]:
    if _git_commit() != expected_analysis_commit:
        raise RuntimeError("analysis commit mismatch")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT_DIR, check=True,
        capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("analysis requires a clean checkout")
    source_hashes = design["source_files"]
    for name, expected in source_hashes.items():
        path = source_root / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"long-budget source hash mismatch: {name}")
    short = design["short_budget_reference"]
    short_hashes = {
        "run_manifest.json": short["run_manifest_sha256"],
        "evaluation_rows.csv": short["evaluation_rows_sha256"],
        "training_signal_rows.csv": short["training_signal_rows_sha256"],
        "checkpoint_selection.json": short["checkpoint_selection_sha256"],
    }
    for name, expected in short_hashes.items():
        path = short_root / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"short-budget source hash mismatch: {name}")
    manifest = _load_json(source_root / "run_manifest.json")
    receipt = _load_json(source_root / "completion_receipt.json")
    if (
        source_root.name != design["scientific_run_id"]
        or short_root.name != short["run_id"]
        or manifest.get("git_commit") != design["scientific_commit"]
        or receipt.get("status") != "complete"
        or receipt.get("learned_cells") != 20
        or receipt.get("learned_environment_steps") != 115200
        or receipt.get("learned_optimizer_steps") != 15360
        or receipt.get("evaluation_rows") != 400
        or receipt.get("formal_or_holdout_reads") != 0
    ):
        raise RuntimeError("scientific completion identity drift")
    return {
        "source_inventory_files": _verify_inventory(source_root),
        "source_file_sha256": source_hashes,
        "short_file_sha256": short_hashes,
        "source_manifest_sha256": source_hashes["run_manifest.json"],
        "scientific_commit": design["scientific_commit"],
        "analysis_commit": expected_analysis_commit,
    }


def _aggregate_pair_rows(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]], split: str,
    method: str, seed: str,
) -> dict[str, Any]:
    short_rows = [short for short, _ in pairs]
    long_rows = [long for _, long in pairs]
    record: dict[str, Any] = {
        "split": split, "method": method, "seed": seed, "instances": len(pairs),
        "short_environment_steps": 1440, "long_environment_steps": 5760,
        "comparison": "budget_plus_proportionally_scaled_selection_schedule",
    }
    for metric in PAIR_METRICS:
        short_value, long_value = _mean(short_rows, metric), _mean(long_rows, metric)
        record[f"short_{metric}"] = short_value
        record[f"long_{metric}"] = long_value
        record[f"delta_{metric}"] = long_value - short_value
    return record


def _prefix_identity(
    source_root: Path, short_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    short_signals = _read_csv(short_root / "training_signal_rows.csv")
    long_signals = [
        row for row in _read_csv(source_root / "training_signal_rows.csv")
        if int(row["global_cell_step"]) <= 1440
    ]
    pairs = _pair_rows(short_signals, long_signals, SIGNAL_KEYS)
    by_cell: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for short, long in pairs:
        by_cell[(short["method"], short["seed"])].append((short, long))
    prefix_rows = []
    for (method, seed), cell_pairs in sorted(by_cell.items()):
        exact_mismatches = sum(short != long for short, long in cell_pairs)
        trajectory_mismatches = sum(
            any(short[field] != long[field] for field in TRAJECTORY_FIELDS)
            for short, long in cell_pairs
        )
        short_rows = [short for short, _ in cell_pairs]
        long_rows = [long for _, long in cell_pairs]
        prefix_rows.append({
            "method": method, "seed": seed, "prefix_steps": len(cell_pairs),
            "short_prefix_sha256": _canonical_rows_sha256(short_rows, SIGNAL_KEYS),
            "long_prefix_sha256": _canonical_rows_sha256(long_rows, SIGNAL_KEYS),
            "exact_row_mismatches": exact_mismatches,
            "trajectory_mismatches": trajectory_mismatches,
            "exact_prefix_match": exact_mismatches == 0,
            "trajectory_match": trajectory_mismatches == 0,
            "interpretation": (
                "identical_training_prefix"
                if exact_mismatches == 0
                else "same_trajectory_different_learning_state"
                if trajectory_mismatches == 0
                else "trajectory_diverged_possible_rng_or_dev_evaluation_timing"
            ),
        })

    short_candidates = _load_json(short_root / "checkpoint_selection.json")
    long_candidates = _load_json(source_root / "checkpoint_selection.json")
    short_24 = [row for row in short_candidates if int(row["update_index"]) == 24]
    long_24 = [row for row in long_candidates if int(row["update_index"]) == 24]
    checkpoint_pairs = _pair_rows(short_24, long_24, ("method", "seed", "update_index"))
    checkpoint_rows = []
    for short, long in checkpoint_pairs:
        short_path = short_root / short["checkpoint"]
        long_path = source_root / long["checkpoint"]
        short_actual, long_actual = _sha256(short_path), _sha256(long_path)
        if short_actual != short["checkpoint_sha256"] or long_actual != long["checkpoint_sha256"]:
            raise RuntimeError("update-24 checkpoint file hash mismatch")
        checkpoint_rows.append({
            "method": short["method"], "seed": short["seed"], "update_index": 24,
            "short_checkpoint_sha256": short_actual,
            "long_checkpoint_sha256": long_actual,
            "checkpoint_bytes_identical": short_actual == long_actual,
        })
    return prefix_rows, checkpoint_rows


def analyze(
    source_root: Path,
    short_root: Path,
    analysis_root: Path,
    *,
    config_path: Path,
    expected_analysis_commit: str,
) -> dict[str, Any]:
    if analysis_root.exists():
        raise FileExistsError(f"create-only analysis root already exists: {analysis_root}")
    design = _load_json(config_path)
    if design.get("schema_version") != "calibrated_workflow_strong_baselines_budget_extension_analysis_v1":
        raise RuntimeError("analysis protocol identity drift")
    if tuple(design["pairing_keys"]) != PAIR_KEYS:
        raise RuntimeError("pairing key protocol drift")
    identity = _validate_frozen_sources(source_root, short_root, design, expected_analysis_commit)
    manifest = _load_json(source_root / "run_manifest.json")
    receipt = _load_json(source_root / "completion_receipt.json")
    long_rows = _read_csv(source_root / "evaluation_rows.csv")
    short_rows = [
        row for row in _read_csv(short_root / "evaluation_rows.csv")
        if row["method"] in LEARNED_METHODS
    ]
    pairs = _pair_rows(short_rows, long_rows)
    if len(pairs) != 400:
        raise RuntimeError(f"paired learned evaluation count drift: {len(pairs)}")

    paired_instance_rows = []
    grouped_pairs: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for short, long in pairs:
        grouped_pairs[(short["split"], short["method"], short["seed"])].append((short, long))
        row: dict[str, Any] = {field: short[field] for field in PAIR_KEYS}
        row["both_completed"] = (
            float(short["workflow_completion_rate"]) == 1.0
            and float(long["workflow_completion_rate"]) == 1.0
        )
        for metric in PAIR_METRICS:
            short_value, long_value = float(short[metric]), float(long[metric])
            row[f"short_{metric}"] = short_value
            row[f"long_{metric}"] = long_value
            row[f"delta_{metric}"] = long_value - short_value
        paired_instance_rows.append(row)

    paired_seed_rows = []
    for (split, method, seed), cell_pairs in sorted(grouped_pairs.items()):
        paired_seed_rows.append(_aggregate_pair_rows(cell_pairs, split, method, seed))
    for method in LEARNED_METHODS:
        for seed in ("7", "17", "29", "43", "61"):
            combined = [pair for key, rows in grouped_pairs.items() if key[1:] == (method, seed) for pair in rows]
            paired_seed_rows.append(_aggregate_pair_rows(combined, "combined_development", method, seed))

    metric_fields = [field for field in paired_seed_rows[0] if field.startswith(("short_", "long_", "delta_"))]
    method_rows = []
    for (split, method), group in sorted(_group(paired_seed_rows, ("split", "method")).items()):
        completion_deltas = [float(row["delta_workflow_completion_rate"]) for row in group]
        on_time_deltas = [float(row["delta_on_time_workflow_completion_rate"]) for row in group]
        failure_deltas = [float(row["delta_service_failure_rate"]) for row in group]
        method_rows.append({
            "split": split, "method": method, "seeds": len(group),
            **{f"mean_{field}": _mean(group, field) for field in metric_fields},
            "completion_positive_zero_negative_seeds": f"{sum(v > 0 for v in completion_deltas)}/{sum(v == 0 for v in completion_deltas)}/{sum(v < 0 for v in completion_deltas)}",
            "on_time_positive_zero_negative_seeds": f"{sum(v > 0 for v in on_time_deltas)}/{sum(v == 0 for v in on_time_deltas)}/{sum(v < 0 for v in on_time_deltas)}",
            "failure_lower_zero_higher_seeds": f"{sum(v < 0 for v in failure_deltas)}/{sum(v == 0 for v in failure_deltas)}/{sum(v > 0 for v in failure_deltas)}",
            "interpretation_scope": "development_only_seed_descriptive_no_independent_clusters",
        })

    common_instance_rows = []
    conditional_rows = []
    for (split, method, seed), cell_pairs in sorted(grouped_pairs.items()):
        common = [pair for pair in cell_pairs if float(pair[0]["workflow_completion_rate"]) == float(pair[1]["workflow_completion_rate"]) == 1.0]
        for short, long in common:
            metrics = (
                "completed_sample_elapsed_seconds", "total_transfer_mb", "model_prepare_mb",
                "state_transfer_mb", "input_transfer_mb", "recompute_seconds",
            )
            common_instance_rows.append({
                **{field: short[field] for field in PAIR_KEYS},
                **{f"short_{metric}": float(short[metric]) for metric in metrics},
                **{f"long_{metric}": float(long[metric]) for metric in metrics},
                **{f"delta_{metric}": float(long[metric]) - float(short[metric]) for metric in metrics},
            })
        short_completed = [short for short, _ in cell_pairs if float(short["workflow_completion_rate"]) == 1.0]
        long_completed = [long for _, long in cell_pairs if float(long["workflow_completion_rate"]) == 1.0]
        conditional_rows.append({
            "split": split, "method": method, "seed": seed, "instances": len(cell_pairs),
            "short_completion_coverage": len(short_completed) / len(cell_pairs),
            "long_completion_coverage": len(long_completed) / len(cell_pairs),
            "common_completed_instances": len(common),
            "common_completed_coverage": len(common) / len(cell_pairs),
            "short_elapsed_common_mean": mean(float(short["completed_sample_elapsed_seconds"]) for short, _ in common) if common else None,
            "long_elapsed_common_mean": mean(float(long["completed_sample_elapsed_seconds"]) for _, long in common) if common else None,
            "long_minus_short_elapsed_common_mean": mean(float(long["completed_sample_elapsed_seconds"]) - float(short["completed_sample_elapsed_seconds"]) for short, long in common) if common else None,
            "survivor_warning": "interpret_only_with_all_instance_paired_table_and_coverage",
        })

    training = _load_json(source_root / "training_summary.json")
    candidates = _load_json(source_root / "checkpoint_selection.json")
    training_by_cell = {(row["method"], str(row["seed"])): row for row in training}
    selected_candidate_by_cell = {}
    for candidate in candidates:
        key = (candidate["method"], str(candidate["seed"]))
        if int(candidate["update_index"]) == int(training_by_cell[key]["selected_update"]):
            if key in selected_candidate_by_cell:
                raise RuntimeError(f"duplicate selected candidate: {key}")
            selected_candidate_by_cell[key] = candidate
    if set(selected_candidate_by_cell) != set(training_by_cell):
        raise RuntimeError("selected candidate identity drift")

    selected_rows = []
    selection_rows = []
    long_groups = _group(long_rows, ("split", "method", "seed"))
    for (split, method, seed), group in sorted(long_groups.items()):
        selection = training_by_cell[(method, seed)]
        candidate = selected_candidate_by_cell[(method, seed)]
        dev = candidate["dev_rows"]
        selected_rows.append({
            "split": split, "method": method, "seed": seed,
            "selected_update": selection["selected_update"],
            "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
            **{metric: _mean(group, metric) for metric in PAIR_METRICS},
            "role": "selected_checkpoint_exposed_development_diagnostic",
        })
        selection_rows.append({
            "method": method, "seed": seed, "selected_update": selection["selected_update"],
            "evaluation_split": split, "selection_score": json.dumps(candidate["score"], separators=(",", ":")),
            "dev_workflow_completion_rate": _mean(dev, "workflow_completion_rate"),
            "evaluation_workflow_completion_rate": _mean(group, "workflow_completion_rate"),
            "dev_on_time_workflow_completion_rate": _mean(dev, "on_time_workflow_completion_rate"),
            "evaluation_on_time_workflow_completion_rate": _mean(group, "on_time_workflow_completion_rate"),
            "dev_minus_evaluation_completion": _mean(dev, "workflow_completion_rate") - _mean(group, "workflow_completion_rate"),
            "evaluation_used_for_selection": False,
            "checkpoint_reselection_performed": False,
        })

    fixed = _read_csv(source_root / "fixed_endpoint_dev_rows.csv")
    fixed_rows = []
    for (method, seed, update), group in sorted(_group(fixed, ("method", "seed", "endpoint_update")).items()):
        fixed_rows.append({
            "method": method, "seed": seed, "endpoint_update": update,
            "dev_instances": len(group),
            **{metric: _mean(group, metric) for metric in PAIR_METRICS},
            "role": "fixed_dev_diagnostic_not_selected_checkpoint_table",
        })

    selection_curve = []
    for candidate in candidates:
        selection = training_by_cell[(candidate["method"], str(candidate["seed"]))]
        selection_curve.append({
            "method": candidate["method"], "seed": candidate["seed"],
            "update_index": candidate["update_index"],
            "environment_steps": int(candidate["update_index"]) * 60,
            "selected": int(candidate["update_index"]) == int(selection["selected_update"]),
            "selection_score": json.dumps(candidate["score"], separators=(",", ":")),
            **{metric: _mean(candidate["dev_rows"], metric) for metric in PAIR_METRICS},
        })

    optimizer = _read_csv(source_root / "optimizer_step_records.csv")
    optimizer_curve = []
    for (method, seed, update), group in sorted(_group(optimizer, ("method", "seed", "update_index")).items()):
        optimizer_curve.append({
            "method": method, "seed": seed, "update_index": update,
            "environment_steps": int(update) * 60, "optimizer_steps": len(group),
            **{metric: _mean(group, metric) for metric in OPTIMIZER_METRICS},
        })
    training_cost_rows = [{
        "method": row["method"], "seed": row["seed"],
        "parameter_count": row["parameter_count"], "environment_steps": row["environment_steps"],
        "update_opportunities": row["update_opportunities"], "optimizer_steps": row["optimizer_steps"],
        "elapsed_seconds": row["elapsed_seconds"], "selected_update": row["selected_update"],
    } for row in training]

    prefix_rows, checkpoint_24_rows = _prefix_identity(source_root, short_root)
    prefix_exact = all(row["exact_prefix_match"] for row in prefix_rows)
    checkpoints_exact = all(row["checkpoint_bytes_identical"] for row in checkpoint_24_rows)
    reused = _load_json(source_root / "reused_nonlearned_reference.json")
    if reused.get("rows") != 40 or reused.get("reuse_mode") != "read_only_reference_no_reevaluation":
        raise RuntimeError("nonlearned reference identity drift")

    sa_combined = next(row for row in method_rows if row["split"] == "combined_development" and row["method"] == "sa_ghmappo")
    result = {
        "schema_version": "causal_strong_baseline_budget_extension_analysis_v2",
        "analysis_run_id": design["analysis_run_id"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "literature_cutoff": "2026-10-09", "target_venue": "IEEE TMC",
        "policy_version": "docs/project/top_journal_review_policy.md@d25ebcd",
        "evidence_level": "L2_complete_development_artifact_no_independent_test",
        **identity,
        "scientific_receipt": receipt,
        "source_manifest_training_budget": manifest["training_budget"],
        "paired_all_instance_rows": len(paired_instance_rows),
        "common_completed_rows": len(common_instance_rows),
        "fixed_endpoint_dev_rows": len(fixed),
        "training_prefix_exact": prefix_exact,
        "update24_checkpoint_bytes_identical": checkpoints_exact,
        "sa_combined_development": sa_combined,
        "nonlearned_reference": reused,
        "automatic_algorithm_followup": False,
        "claim_boundary": design["claim_boundary"],
        "limitations": [
            "all_36_instances_previously_consumed_development",
            "budget_and_proportionally_scaled_checkpoint_schedule_change_jointly",
            "same_source_window_repeated_across_seeds_not_independent_clusters",
            "selected_checkpoint_checks_are_exposed_development_not_selection_inputs",
            "two_step_and_popularity_reused_without_reevaluation",
        ],
    }

    analysis_root.mkdir(parents=True)
    _write_csv(analysis_root / "paired_all_instance_diagnostics.csv", paired_instance_rows)
    _write_csv(analysis_root / "paired_method_seed_diagnostics.csv", paired_seed_rows)
    _write_csv(analysis_root / "method_budget_effect_summary.csv", method_rows)
    _write_csv(analysis_root / "common_completed_instance_diagnostics.csv", common_instance_rows)
    _write_csv(analysis_root / "conditional_completed_summary.csv", conditional_rows)
    _write_csv(analysis_root / "selected_checkpoint_service_diagnostics.csv", selected_rows)
    _write_csv(analysis_root / "selection_vs_exposed_development.csv", selection_rows)
    _write_csv(analysis_root / "fixed_endpoint_dev_diagnostics.csv", fixed_rows)
    _write_csv(analysis_root / "checkpoint_selection_curve.csv", selection_curve)
    _write_csv(analysis_root / "optimizer_update_diagnostics.csv", optimizer_curve)
    _write_csv(analysis_root / "training_cost_diagnostics.csv", training_cost_rows)
    _write_csv(analysis_root / "training_prefix_identity.csv", prefix_rows)
    _write_csv(analysis_root / "update24_checkpoint_identity.csv", checkpoint_24_rows)
    _write_json(analysis_root / "reused_nonlearned_reference.json", reused)
    _write_json(analysis_root / "analysis_summary.json", result)
    _write_json(analysis_root / "analysis_manifest.json", {
        "schema_version": design["schema_version"], "analysis_run_id": design["analysis_run_id"],
        "created_at": result["reviewed_at"], "analysis_commit": expected_analysis_commit,
        "scientific_commit": design["scientific_commit"], "source_run_id": design["scientific_run_id"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "config": {"path": str(config_path.relative_to(ROOT_DIR)), "sha256": _sha256(config_path)},
        "command": list(sys.argv), "new_training_steps": 0, "new_evaluation_rows": 0,
    })
    _write_json(analysis_root / "analysis_completion_receipt.json", {
        "status": "complete", "paired_all_instance_rows": len(paired_instance_rows),
        "paired_method_seed_rows": len(paired_seed_rows), "common_completed_rows": len(common_instance_rows),
        "fixed_endpoint_dev_rows": len(fixed), "optimizer_update_rows": len(optimizer_curve),
        "training_prefix_cells": len(prefix_rows), "update24_checkpoint_cells": len(checkpoint_24_rows),
        "new_training_steps": 0, "new_evaluation_rows": 0, "formal_or_holdout_reads": 0,
    })
    _integrity(analysis_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--short_budget_root", required=True)
    parser.add_argument("--analysis_root", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected_analysis_commit", required=True)
    args = parser.parse_args()
    result = analyze(
        Path(args.source_root).resolve(), Path(args.short_budget_root).resolve(),
        Path(args.analysis_root).resolve(), config_path=(ROOT_DIR / args.config).resolve(),
        expected_analysis_commit=args.expected_analysis_commit,
    )
    print(json.dumps({"status": "complete", "analysis_run_id": result["analysis_run_id"]}))


if __name__ == "__main__":
    main()
