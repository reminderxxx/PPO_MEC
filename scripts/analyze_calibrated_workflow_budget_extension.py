"""Analyze the preregistered 4x learned-method budget extension."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json  # noqa: E402
from scripts.run_calibrated_workflow_strong_baselines import (  # noqa: E402
    BUDGET_EXTENSION_VERSION,
    LEARNED_METHODS,
    _load_json,
    _read_csv,
    _validate_short_budget_reference,
)


SERVICE_METRICS = (
    "on_time_workflow_completion_rate",
    "workflow_completion_rate",
    "service_failure_rate",
    "service_failures",
    "failed_service_attempt_seconds_proxy",
    "unfinished_after_deadline_rate",
    "node_coverage_rate",
    "modeled_completion_seconds",
    "total_transfer_mb",
    "model_prepare_mb",
    "recompute_seconds",
    "invalid_prepare_attempts",
    "max_consecutive_no_progress_steps",
)

OPTIMIZER_METRICS = (
    "actor_loss",
    "env_action_ppo_loss",
    "value_loss",
    "auxiliary_loss",
    "total_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "policy_grad_norm",
    "weighted_value_grad_norm",
    "weighted_auxiliary_grad_norm",
    "pre_clip_total_grad_norm",
    "post_clip_total_grad_norm_upper_bound",
    "global_clip_scale",
)


def _mean(group: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in group)


def _group(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    return grouped


def _verify_inventory(root: Path) -> None:
    inventory = _load_json(root / "artifact_integrity.json")
    for item in inventory["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"scientific artifact integrity mismatch: {item['path']}")


def analyze(root: Path, short_root: Path) -> dict[str, Any]:
    _verify_inventory(root)
    receipt = _load_json(root / "completion_receipt.json")
    manifest = _load_json(root / "run_manifest.json")
    design_path = ROOT_DIR / manifest["design_config"]["path"]
    design = _load_json(design_path)
    if design["schema_version"] != BUDGET_EXTENSION_VERSION:
        raise RuntimeError("budget-extension design identity drift")
    if _sha256(design_path) != manifest["design_config"]["sha256"]:
        raise RuntimeError("budget-extension design hash mismatch")
    short_reference = _validate_short_budget_reference(design, short_root)
    embedded = _load_json(root / "short_budget_reference.json")
    if embedded["file_sha256"] != short_reference["file_sha256"]:
        raise RuntimeError("embedded short-budget reference drift")
    if (
        receipt.get("status") != "complete"
        or receipt.get("learned_cells") != 20
        or receipt.get("learned_environment_steps") != 115200
        or receipt.get("learned_optimizer_steps") != 15360
        or receipt.get("evaluation_rows") != 400
        or receipt.get("formal_or_holdout_reads") != 0
    ):
        raise RuntimeError("long-budget completion identity drift")
    if (
        manifest.get("interface_profile") != "calibrated_workflow_interface_v3_prefix_only"
        or manifest.get("reward_profile") != "original_reward_v1"
        or manifest.get("critic_target_normalization") != "raw_disabled"
        or tuple(manifest.get("learned_methods", [])) != LEARNED_METHODS
        or manifest.get("seeds") != [7, 17, 29, 43, 61]
        or manifest.get("causal_implementation_commit") != "a08869388f9962149198054b5f8a0d6dd4d07eae"
        or manifest.get("nonlearned_evaluation_mode") != "reused_short_budget_reference_no_reevaluation"
    ):
        raise RuntimeError("long-budget scientific identity drift")

    long_rows = _read_csv(root / "evaluation_rows.csv")
    short_rows = [row for row in _read_csv(short_root / "evaluation_rows.csv") if row["method"] in LEARNED_METHODS]
    ledger = _read_csv(root / "behavior_ledger.csv")
    signals = _read_csv(root / "training_signal_rows.csv")
    optimizer = _read_csv(root / "optimizer_step_records.csv")
    fixed = _read_csv(root / "fixed_endpoint_dev_rows.csv")
    training = _load_json(root / "training_summary.json")
    candidates = _load_json(root / "checkpoint_selection.json")
    endpoint_checkpoints = _load_json(root / "fixed_endpoint_checkpoint_identity.json")
    expected_ids = {
        (split, method, str(seed), design_id)
        for split, ids in (
            ("regression", [f"regression_{index:02d}" for index in range(12)]),
            ("frozen_check", [f"frozen_check_{index:02d}" for index in range(8)]),
        )
        for method in LEARNED_METHODS
        for seed in (7, 17, 29, 43, 61)
        for design_id in ids
    }
    long_ids = {(row["split"], row["method"], row["seed"], row["design_id"]) for row in long_rows}
    short_ids = {(row["split"], row["method"], row["seed"], row["design_id"]) for row in short_rows}
    if len(long_rows) != 400 or long_ids != expected_ids or short_ids != expected_ids:
        raise RuntimeError("paired learned evaluation identity drift")
    if (
        len(training) != 20
        or len(signals) != 115200
        or len(optimizer) != 15360
        or len(candidates) != 80
        or len(fixed) != 160
        or len(endpoint_checkpoints) != 40
        or not ledger
    ):
        raise RuntimeError("long-budget diagnostic artifact count drift")
    if any(int(row["environment_steps"]) != 5760 or int(row["optimizer_steps"]) != 768 for row in training):
        raise RuntimeError("long-budget learned cell drift")
    for row in training:
        checkpoint = root / row["selected_checkpoint"]
        if not checkpoint.is_file() or _sha256(checkpoint) != row["selected_checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint hash mismatch")

    short_by_id = {(row["split"], row["method"], row["seed"], row["design_id"]): row for row in short_rows}
    paired_cells = []
    for key, group in sorted(_group(long_rows, ("split", "method", "seed")).items()):
        split, method, seed = key
        short_group = [short_by_id[(split, method, seed, row["design_id"])] for row in group]
        record: dict[str, Any] = {
            "split": split,
            "method": method,
            "seed": seed,
            "instances": len(group),
            "short_environment_steps": 1440,
            "long_environment_steps": 5760,
            "comparison": "budget_plus_proportionally_scaled_selection_schedule",
        }
        for metric in SERVICE_METRICS:
            short_value, long_value = _mean(short_group, metric), _mean(group, metric)
            record[f"short_{metric}"] = short_value
            record[f"long_{metric}"] = long_value
            record[f"delta_{metric}"] = long_value - short_value
        paired_cells.append(record)
    _write_csv(root / "budget_paired_seed_diagnostics.csv", paired_cells)

    method_rows = []
    for (split, method), group in sorted(_group(paired_cells, ("split", "method")).items()):
        method_rows.append({
            "split": split,
            "method": method,
            "seeds": len(group),
            **{f"mean_{field}": _mean(group, field) for field in paired_cells[0] if field.startswith(("short_", "long_", "delta_"))},
            "interpretation_scope": "development_only_descriptive_no_significance_claim",
        })
    _write_csv(root / "budget_method_diagnostics.csv", method_rows)

    selected_rows = []
    training_by_cell = {(row["method"], str(row["seed"])): row for row in training}
    for (split, method, seed), group in sorted(_group(long_rows, ("split", "method", "seed")).items()):
        selection = training_by_cell[(method, seed)]
        selected_rows.append({
            "split": split,
            "method": method,
            "seed": seed,
            "selected_update": selection["selected_update"],
            "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
            **{metric: _mean(group, metric) for metric in SERVICE_METRICS},
            "role": "selected_checkpoint_service_diagnostic",
        })
    _write_csv(root / "selected_checkpoint_service_diagnostics.csv", selected_rows)

    fixed_rows = []
    for (method, seed, update), group in sorted(_group(fixed, ("method", "seed", "endpoint_update")).items()):
        fixed_rows.append({
            "method": method,
            "seed": seed,
            "endpoint_update": update,
            "dev_instances": len(group),
            **{metric: _mean(group, metric) for metric in SERVICE_METRICS},
            "role": "fixed_dev_diagnostic_not_checkpoint_selection",
        })
    _write_csv(root / "fixed_endpoint_dev_diagnostics.csv", fixed_rows)

    selection_curve = []
    for candidate in candidates:
        dev_rows = candidate["dev_rows"]
        selection = training_by_cell[(candidate["method"], str(candidate["seed"]))]
        selection_curve.append({
            "method": candidate["method"],
            "seed": candidate["seed"],
            "update_index": candidate["update_index"],
            "environment_steps": int(candidate["update_index"]) * 60,
            "selected": int(candidate["update_index"]) == int(selection["selected_update"]),
            "selection_score": json.dumps(candidate["score"], separators=(",", ":")),
            **{metric: _mean(dev_rows, metric) for metric in SERVICE_METRICS},
        })
    _write_csv(root / "checkpoint_selection_curve.csv", selection_curve)

    optimizer_curve = []
    for (method, seed, update), group in sorted(_group(optimizer, ("method", "seed", "update_index")).items()):
        optimizer_curve.append({
            "method": method,
            "seed": seed,
            "update_index": update,
            "environment_steps": int(update) * 60,
            "optimizer_steps": len(group),
            **{metric: _mean(group, metric) for metric in OPTIMIZER_METRICS},
        })
    _write_csv(root / "optimizer_update_diagnostics.csv", optimizer_curve)

    elapsed_rows = []
    for (split, method, seed), long_group in sorted(_group(long_rows, ("split", "method", "seed")).items()):
        short_group = [short_by_id[(split, method, seed, row["design_id"])] for row in long_group]
        pairs = list(zip(short_group, long_group, strict=True))
        common = [(left, right) for left, right in pairs if float(left["workflow_completion_rate"]) == float(right["workflow_completion_rate"]) == 1.0]
        short_completed = [left for left, _ in pairs if float(left["workflow_completion_rate"]) == 1.0]
        long_completed = [right for _, right in pairs if float(right["workflow_completion_rate"]) == 1.0]
        elapsed_rows.append({
            "split": split,
            "method": method,
            "seed": seed,
            "instances": len(pairs),
            "short_completion_coverage": len(short_completed) / len(pairs),
            "long_completion_coverage": len(long_completed) / len(pairs),
            "common_completed_instances": len(common),
            "common_completed_coverage": len(common) / len(pairs),
            "short_elapsed_common_mean": mean(float(left["completed_sample_elapsed_seconds"]) for left, _ in common) if common else None,
            "long_elapsed_common_mean": mean(float(right["completed_sample_elapsed_seconds"]) for _, right in common) if common else None,
            "long_minus_short_elapsed_common_mean": mean(float(right["completed_sample_elapsed_seconds"]) - float(left["completed_sample_elapsed_seconds"]) for left, right in common) if common else None,
        })
    _write_csv(root / "conditional_elapsed_diagnostics.csv", elapsed_rows)

    sa_rows = [row for row in method_rows if row["method"] == "sa_ghmappo"]
    result = {
        "schema_version": "causal_strong_baseline_budget_extension_analysis_v1",
        "run_id": root.name,
        "git_commit": manifest["git_commit"],
        "short_budget_run_id": short_reference["run_id"],
        "short_budget_file_sha256": short_reference["file_sha256"],
        "intervention": manifest["budget_intervention"],
        "learned_environment_steps": receipt["learned_environment_steps"],
        "learned_optimizer_steps": receipt["learned_optimizer_steps"],
        "paired_selected_rows": len(long_rows),
        "fixed_endpoint_dev_rows": len(fixed),
        "nonlearned_methods": "reused_by_exact_hash_without_reevaluation",
        "sa_budget_effect": sa_rows,
        "automatic_algorithm_followup": False,
        "formal_or_holdout_reads": 0,
        "limitations": [
            "all_36_instances_previously_consumed_development",
            "budget_and_proportionally_scaled_checkpoint_schedule_change_jointly",
            "same_source_window_repeated_across_seeds",
            "descriptive_five_seed_comparison_no_independent_test",
            "two_step_and_popularity_results_reused_from_short_budget_run",
        ],
    }
    _write_json(root / "budget_extension_analysis.json", result)
    _write_json(root / "analysis_receipt.json", {
        "status": "complete",
        "paired_selected_rows": len(long_rows),
        "paired_seed_rows": len(paired_cells),
        "fixed_endpoint_dev_rows": len(fixed),
        "optimizer_update_rows": len(optimizer_curve),
        "short_budget_reference_sha256": short_reference["file_sha256"],
        "formal_or_holdout_reads": 0,
    })
    _integrity(root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--short_budget_root", required=True)
    args = parser.parse_args()
    result = analyze(Path(args.output_root).resolve(), Path(args.short_budget_root).resolve())
    print(json.dumps({"status": "complete", "run_id": result["run_id"]}))


if __name__ == "__main__":
    main()
