"""Deterministic, development-only analysis of the frozen baseline run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json
from scripts.run_calibrated_workflow_strong_baselines import LEARNED_METHODS, HEURISTIC_METHOD, MODEL_BASED_METHOD


METRICS = (
    "on_time_workflow_completion_rate", "workflow_completion_rate", "service_failure_rate",
    "service_failures", "failed_service_attempt_seconds_proxy", "max_consecutive_no_progress_steps",
    "node_coverage_rate", "modeled_completion_seconds", "total_transfer_mb", "model_prepare_mb",
    "state_transfer_mb", "input_transfer_mb", "recompute_seconds", "decision_overhead_ms",
    "migration_attempts", "migration_successes", "invalid_prepare_attempts",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(root: Path) -> dict:
    inventory = json.loads((root / "artifact_integrity.json").read_text())
    for item in inventory["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"scientific artifact integrity mismatch: {item['path']}")
    receipt = json.loads((root / "completion_receipt.json").read_text())
    manifest = json.loads((root / "run_manifest.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("learned_environment_steps") != 28800:
        raise RuntimeError("scientific run is incomplete or step budget drifted")
    if manifest.get("interface_profile") != "calibrated_workflow_interface_v3_prefix_only":
        raise RuntimeError("causal interface identity drift")
    rows = _read_csv(root / "evaluation_rows.csv")
    ledger = _read_csv(root / "behavior_ledger.csv")
    signals = _read_csv(root / "training_signal_rows.csv")
    training = json.loads((root / "training_summary.json").read_text())
    candidates = json.loads((root / "checkpoint_selection.json").read_text())
    optimizer = _read_csv(root / "optimizer_step_records.csv")
    expected_methods = set(LEARNED_METHODS) | {HEURISTIC_METHOD, MODEL_BASED_METHOD}
    expected = {
        (split, method, str(seed), design_id)
        for split, ids in (("regression", [f"regression_{i:02d}" for i in range(12)]), ("frozen_check", [f"frozen_check_{i:02d}" for i in range(8)]))
        for method in expected_methods
        for seed in (manifest["seeds"] if method in LEARNED_METHODS else ["rule"])
        for design_id in ids
    }
    actual = {(r["split"], r["method"], r["seed"], r["design_id"]) for r in rows}
    if len(rows) != 440 or actual != expected:
        raise RuntimeError(f"evaluation identity/count failure: {len(rows)} rows, {len(actual ^ expected)} mismatches")
    if len(training) != 20 or len(signals) != 28800 or len(candidates) != 80 or len(optimizer) != 3840 or not ledger:
        raise RuntimeError("training or behavior artifact count failure")
    if any(int(r["environment_steps"]) != 1440 or int(r["optimizer_steps"]) != 192 for r in training):
        raise RuntimeError("learned cell budget failure")
    for row in training:
        path = root / row["selected_checkpoint"]
        if not path.is_file() or _sha256(path) != row["selected_checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint hash mismatch")
    prediction = defaultdict(lambda: {"known": 0, "unknown": 0, "correct_next": 0, "wrong_next": 0})
    for row in ledger:
        provenance = json.loads(row["prediction_provenance"])
        if provenance.get("predictor_sha256") != manifest["source_interval_validation"]["causal_predictor"]["full_model_sha256"]:
            raise RuntimeError("evaluation predictor hash mismatch")
        if int(provenance["prefix_end_index"]) != int(row["step_index"]):
            raise RuntimeError("prediction prefix boundary mismatch")
        cell = prediction[(row["split"], row["method"], row["seed"])]
        if provenance["status"] == "unknown":
            cell["unknown"] += 1
        else:
            cell["known"] += 1
            cell["correct_next" if row["predicted_next_rsu_id"] == row["actual_next_rsu_id"] else "wrong_next"] += 1
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"], row["seed"])].append(row)
    summary = []
    by_id = {(r["split"], r["method"], r["seed"], r["design_id"]): r for r in rows}
    for (split, method, seed), group in sorted(groups.items()):
        completed = [r for r in group if float(r["workflow_completion_rate"]) == 1.0]
        summary.append({
            "split": split, "method": method, "seed": seed, "instances": len(group),
            **{name: mean(float(r[name]) for r in group) for name in METRICS},
            "completed_sample_elapsed_coverage": len(completed) / len(group),
            "completed_sample_elapsed_seconds": mean(float(r["completed_sample_elapsed_seconds"]) for r in completed) if completed else None,
            "prediction": prediction[(split, method, seed)],
            "status": "development_only_no_independent_test",
        })
    paired = []
    for row in rows:
        if row["method"] not in LEARNED_METHODS:
            continue
        rule = by_id[(row["split"], MODEL_BASED_METHOD, "rule", row["design_id"])]
        both = float(row["workflow_completion_rate"]) == float(rule["workflow_completion_rate"]) == 1.0
        paired.append({
            "split": row["split"], "method": row["method"], "seed": row["seed"],
            "design_id": row["design_id"], "both_completed": both,
            "elapsed_seconds_difference_vs_two_step": float(row["completed_sample_elapsed_seconds"]) - float(rule["completed_sample_elapsed_seconds"]) if both else None,
            "on_time_difference_vs_two_step": float(row["on_time_workflow_completion_rate"]) - float(rule["on_time_workflow_completion_rate"]),
            "transfer_mb_difference_vs_two_step": float(row["total_transfer_mb"]) - float(rule["total_transfer_mb"]),
            "capability_warning": "two_step_uses_exact_transition_clone",
        })
    result = {
        "schema_version": "causal_strong_baseline_development_analysis_v1",
        "run_id": root.name, "git_commit": manifest["git_commit"],
        "evaluation_rows": len(rows), "training_signal_rows": len(signals),
        "learned_environment_steps": receipt["learned_environment_steps"],
        "formal_or_holdout_reads": 0, "independent_test": False,
        "summary": summary, "paired_instance_rows": paired,
        "training_cost": training,
        "optimizer_step_rows": len(optimizer), "checkpoint_candidates": len(candidates),
        "limitations": ["all_36_instances_previously_consumed_development", "same_source_window_repeated_across_seeds", "two_step_exact_transition_model_privilege", "no_real_wireless_or_model_inference"],
    }
    _write_json(root / "development_analysis.json", result)
    _write_json(root / "analysis_receipt.json", {"status": "complete", "evaluation_rows": len(rows), "paired_rows": len(paired), "prediction_audited_steps": len(ledger)})
    _integrity(root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    result = analyze(Path(args.output_root).resolve())
    print(json.dumps({"status": "complete", "run_id": result["run_id"], "evaluation_rows": result["evaluation_rows"]}))


if __name__ == "__main__":
    main()
