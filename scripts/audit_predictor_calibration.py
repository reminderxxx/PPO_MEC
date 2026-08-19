"""Audit and materialize the G12 causal calibrated predictor contract.

This entry point never trains a predictor and never reads RL rewards.  It splits
the frozen non-hidden NGSIM training material into predictor-train and
calibration windows, keeps the existing dev windows for evaluation, fits only
the binary handoff temperature, and emits a self-contained audit bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import build_selected_workflow_states
from src.evaluators.real_sample_support import load_real_mobility_bundle
from src.metrics.recorder import EpisodeRecorder
from src.predictors.calibration import (
    CALIBRATION_AUDIT_VERSION,
    RELIABILITY_BIN_CONTRACT_VERSION,
    TEMPERATURE_SCALING_VERSION,
    abstention_reason_counts,
    apply_binary_temperature,
    audit_predictor_splits,
    audit_vehicle_group_overlap,
    binary_calibration_metrics,
    canonical_sha256,
    eta_regression_metrics,
    fit_binary_temperature,
    identity_calibration,
    risk_coverage_curve,
    select_abstention_threshold,
    select_calibration_method,
    sigmoid,
)
from src.predictors.causal_snapshot import (
    CALIBRATION_ARTIFACT_CONTRACT_VERSION,
    CAUSAL_PREDICTOR_SNAPSHOT_CONTRACT_VERSION,
    build_causal_predictor_snapshot,
    consume_snapshot,
    load_calibration_artifact,
    sha256_file,
    staleness_diagnostics,
    validate_causal_predictor_snapshot,
)


DEFAULT_PREDICTOR_ROOT = ROOT_DIR / "artifacts" / "experiments" / "top_journal_v112_predictor_training_20260809" / "supervised_handoff_predictor_20260809_033406_140764"
DEFAULT_WINDOW_ROOT = ROOT_DIR / "configs" / "experiment" / "top_journal_v71_strict_split_20260730"
DEFAULT_AUDIT_DOC = ROOT_DIR / "docs" / "project" / "predictor_causality_calibration_audit_20260819.md"
CONTRACT_DOC = ROOT_DIR / "docs" / "project" / "causal_calibrated_predictor_snapshot_contract.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit G12 causal calibrated predictor snapshots")
    parser.add_argument("--quality_rows_path", default=str(DEFAULT_PREDICTOR_ROOT / "predictor_quality_rows.csv"))
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_PREDICTOR_ROOT / "supervised_handoff_predictor.pt"))
    parser.add_argument("--predictor_manifest_path", default=str(DEFAULT_PREDICTOR_ROOT / "predictor_metrics_manifest.json"))
    parser.add_argument("--train_window_plan_path", default=str(DEFAULT_WINDOW_ROOT / "train_window_plan.json"))
    parser.add_argument("--evaluation_window_plan_path", default=str(DEFAULT_WINDOW_ROOT / "dev_window_plan.json"))
    parser.add_argument("--workflow_csv_path", default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--mobility_csv_path", default="")
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--real_trace_steps", type=int, default=5)
    parser.add_argument("--skip_real_trace", action="store_true")
    parser.add_argument("--minimum_coverage", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--run_id", default="causal_predictor_snapshot_validation_20260819_g12_v1")
    parser.add_argument("--output_root", default=str(ROOT_DIR / "artifacts" / "analysis"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def logit(probability: float) -> float:
    clipped = min(max(float(probability), 1.0e-12), 1.0 - 1.0e-12)
    return math.log(clipped / (1.0 - clipped))


def partition_windows(train_windows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Frozen before metrics: every fourth train-plan window is calibration."""
    calibration = [window for index, window in enumerate(train_windows) if index % 4 == 3]
    predictor_train = [window for index, window in enumerate(train_windows) if index % 4 != 3]
    return predictor_train, calibration


def hard_classification_metrics(
    rows: list[dict[str, str]],
    *,
    label_key: str,
    prediction_key: str,
    class_names: list[str],
    empty_label_class: str | None = None,
) -> dict[str, Any]:
    name_to_index = {name: index for index, name in enumerate(class_names)}
    pairs: list[tuple[int, int]] = []
    unknown = 0
    empty_label_count = 0
    for row in rows:
        label_name = row.get(label_key, "")
        if label_name == "":
            empty_label_count += 1
            if empty_label_class is not None:
                label_name = empty_label_class
        try:
            prediction = int(row.get(prediction_key, ""))
        except ValueError:
            unknown += 1
            continue
        if label_name not in name_to_index or not 0 <= prediction < len(class_names):
            unknown += 1
            continue
        pairs.append((name_to_index[label_name], prediction))
    confusion = [[0 for _ in class_names] for _ in class_names]
    for label, prediction in pairs:
        confusion[label][prediction] += 1
    f1_rows: list[tuple[float, int]] = []
    for class_index in range(len(class_names)):
        true_positive = sum(label == class_index and prediction == class_index for label, prediction in pairs)
        false_positive = sum(label != class_index and prediction == class_index for label, prediction in pairs)
        false_negative = sum(label == class_index and prediction != class_index for label, prediction in pairs)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = 2.0 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
        support = sum(label == class_index for label, _ in pairs)
        f1_rows.append((f1, support))
    count = len(pairs)
    return {
        "availability": "hard_prediction_only" if count else "unavailable",
        "sample_count": count,
        "top_1_accuracy": sum(label == prediction for label, prediction in pairs) / count if count else None,
        "macro_f1": sum(value for value, _ in f1_rows) / len(f1_rows) if f1_rows else None,
        "weighted_f1": sum(value * support for value, support in f1_rows) / count if count else None,
        "confusion_matrix": {"class_names": class_names, "rows": confusion},
        "evaluated_coverage": count / len(rows) if rows else None,
        "empty_label_count": empty_label_count,
        "empty_label_class": empty_label_class,
        "unknown_class_count": unknown,
        "probability_metrics": {
            "multiclass_brier_score": None,
            "negative_log_likelihood": None,
            "top_label_ece": None,
            "classwise_ece": None,
            "top_k_accuracy": None,
            "probability_simplex_validation": None,
        },
        "unavailable_reason": "historical quality rows retain hard class indices but not full multiclass logits",
    }


def build_snapshot_examples(calibration_artifact: dict[str, Any], git_commit: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = dict(
        vehicle_id="synthetic_vehicle",
        predictor_kind="supervised",
        model_identity="synthetic_contract_fixture_not_runtime_model",
        checkpoint_identity={"sha256": "synthetic", "horizon": 3},
        predictor_config_hash=canonical_sha256({"fixture": "g12"}),
        source_dataset_identity={"dataset": "synthetic_contract_fixture", "cross_source": False},
        source_window_plan_identity={"window_id": "synthetic_window"},
        git_commit=git_commit,
        generated_at_step=2,
        generated_at_time=200,
        observation_as_of_step=2,
        observation_as_of_time=200,
        consumed_at_step=2,
        consumed_at_time=200,
        label_horizon=3,
        valid_for_steps=3,
        update_interval_steps=3,
        current_rsu_id="rsu_a",
        class_ids=["rsu_a", "rsu_b", None],
        runtime_rsu_ids=["rsu_a", "rsu_b"],
        raw_next_rsu_logits=[-5.0, 8.0, -6.0],
        raw_handoff_logit=20.0,
        raw_handoff_target_logits=[-5.0, 8.0, -6.0],
        eta_point_estimate=2.0,
        calibration_artifact=calibration_artifact,
        feature_availability_mask={"current_observation": True, "previous_vehicle_position": True},
        normalization_version="supervised_handoff_fixed_clamp_v1",
        history_start_step=1,
        history_end_step=2,
        history_start_time=100,
        history_end_time=200,
        source_frame_interval=[1, 2],
        source_time_interval=[100, 200],
        causal_cutoff_step=2,
        causal_cutoff_time=200,
    )
    accepted = build_causal_predictor_snapshot(**base)
    stale = consume_snapshot(accepted, consumed_at_step=8, consumed_at_time=800)
    cold_args = {**base, "insufficient_history": True, "history_start_step": None, "history_start_time": None}
    cold = build_causal_predictor_snapshot(**cold_args)
    unseen_args = {**base, "runtime_rsu_ids": ["rsu_a"]}
    unseen = build_causal_predictor_snapshot(**unseen_args)
    future = deepcopy(accepted)
    future["causal_time"]["observation_as_of_step"] = 3
    future_validation = validate_causal_predictor_snapshot(future)
    examples = [accepted, stale, cold, unseen]
    return examples, {
        "example_validations": [validate_causal_predictor_snapshot(row) for row in examples],
        "future_as_of_negative_case": future_validation,
        "future_as_of_rejected": future_validation["status"] == "fail",
        "abstention_reason_counts": abstention_reason_counts(examples),
    }


def run_real_trace(
    *, args: argparse.Namespace, calibration_path: Path, checkpoint_path: Path, git_commit: str
) -> dict[str, Any]:
    if args.skip_real_trace:
        return {"availability": "skipped_by_flag"}
    workflow_path = Path(args.workflow_csv_path)
    try:
        mobility_bundle = load_real_mobility_bundle(
            root_dir=ROOT_DIR,
            mobility_source="ngsim",
            mobility_csv_path=args.mobility_csv_path,
            lust_scenario_root="",
            max_mobility_rows=args.max_mobility_rows,
            rsu_layout="auto_dominant_tight",
            frame_offset=0,
            window_length=24,
            window_selector="max_handoff_candidate",
            random_seed=args.random_seed,
        )
        workflows = build_selected_workflow_states(
            workflow_csv_path=workflow_path,
            max_workflows=1,
            workflow_selector="ordered",
            min_tasks=5,
            max_tasks=20,
            random_seed=args.random_seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return {"availability": "unavailable", "reason": str(error), "formal_or_hidden_used": False}
    recorder = EpisodeRecorder(prefetch_validation_window=6)
    manager = PredictorManager(
        predictor_kind="supervised",
        predictor_checkpoint_path=str(checkpoint_path),
        causal_calibrated_snapshot_enabled=True,
        causal_snapshot_calibration_artifact_path=str(calibration_path),
        causal_snapshot_update_interval_steps=3,
        causal_snapshot_valid_for_steps=3,
        causal_snapshot_min_history_steps=1,
        causal_snapshot_fallback_behavior="mask_only",
        causal_snapshot_source_dataset_identity={
            "dataset": "NGSIM",
            "source_path": mobility_bundle.source_path,
            "role": "non_hidden_nonformal_minimal_runtime_trace",
            "cross_source": False,
        },
        causal_snapshot_source_window_plan_identity={
            "window_id": mobility_bundle.rsu_metadata.get("window_id"),
            "frame_interval": [
                mobility_bundle.rsu_metadata.get("frame_offset"),
                mobility_bundle.rsu_metadata.get("frame_offset", 0)
                + mobility_bundle.rsu_metadata.get("window_length", 0)
                - 1,
            ],
            "time_interval": [
                mobility_bundle.rsu_metadata.get("time_index_start"),
                mobility_bundle.rsu_metadata.get("time_index_end"),
            ],
        },
        causal_snapshot_git_commit=git_commit,
    )
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=mobility_bundle.frames),
        workflow_state=workflows[0],
        adapter_catalog=AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"),
        rsu_states=mobility_bundle.rsu_states,
        predictor_manager=manager,
        max_steps=max(args.real_trace_steps + 2, 8),
        mobility_source="ngsim",
    )
    env = GymVecEnv(core_env=core, recorder=recorder)
    recorder.start_episode(
        {
            "evaluation_unit_id": "g12_nonformal_ngsim_minimal",
            "request_replay_fingerprint": None,
            "g08_oracle_contract_version": None,
            "g09_analysis_fingerprint": None,
            "formal": False,
            "hidden": False,
        }
    )
    _, info = env.reset(seed=args.random_seed)
    states: list[dict[str, Any]] = [deepcopy(env.last_semantic_state or {})]
    for _ in range(max(args.real_trace_steps, 1)):
        mask = list(info.get("action_mask", []))
        action = next((index for index, allowed in enumerate(mask) if allowed), 0)
        _, _, terminated, truncated, info = env.step(action)
        states.append(deepcopy(env.last_semantic_state or {}))
        if terminated or truncated:
            break
    summary = recorder.build_summary()
    snapshots = [
        snapshot
        for state in states
        for snapshot in state.get("predictions", {}).get("causal_predictor_snapshots_by_vehicle", {}).values()
    ]
    return {
        "availability": "available",
        "scope": "NGSIM non-hidden non-formal minimal runtime contract validation",
        "formal_or_hidden_used": False,
        "source_path": mobility_bundle.source_path,
        "window_metadata": mobility_bundle.rsu_metadata,
        "step_count": len(summary.get("step_trace", [])),
        "snapshot_count": len(snapshots),
        "accepted_snapshot_count": sum(
            int(snapshot.get("predictions", {}).get("availability_mask", 0)) for snapshot in snapshots
        ),
        "abstention_reason_counts": abstention_reason_counts(snapshots),
        "snapshot_examples": snapshots[:3],
        "decision_observation_trace": summary.get("decision_observation_trace"),
        "predictor_snapshot_provenance": summary.get("predictor_snapshot_provenance"),
        "action_semantics_changed": False,
        "reward_changed": False,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = Path(args.quality_rows_path)
    checkpoint_path = Path(args.checkpoint_path)
    manifest_path = Path(args.predictor_manifest_path)
    train_plan_path = Path(args.train_window_plan_path)
    evaluation_plan_path = Path(args.evaluation_window_plan_path)
    for path in (quality_path, checkpoint_path, manifest_path, train_plan_path, evaluation_plan_path, DEFAULT_AUDIT_DOC):
        if not path.exists():
            raise FileNotFoundError(path)

    git_commit = git_head()
    reviewed_at = datetime.now(timezone.utc).isoformat()
    train_plan = read_json(train_plan_path)
    evaluation_plan = read_json(evaluation_plan_path)
    predictor_train_windows, calibration_windows = partition_windows(train_plan["selected_window_plan"])
    evaluation_windows = list(evaluation_plan["selected_window_plan"])
    overlap = audit_predictor_splits(
        {
            "predictor_train": predictor_train_windows,
            "calibration": calibration_windows,
            "evaluation_dev": evaluation_windows,
        }
    )
    if not overlap["passed"]:
        raise RuntimeError(f"three-way predictor split overlaps: {overlap['overlap_conflicts']}")
    calibration_window_ids = {row["window_id"] for row in calibration_windows}
    predictor_train_window_ids = {row["window_id"] for row in predictor_train_windows}
    evaluation_window_ids = {row["window_id"] for row in evaluation_windows}
    with quality_path.open("r", encoding="utf-8-sig", newline="") as handle:
        quality_rows = list(csv.DictReader(handle))
    rows_by_split = {
        "predictor_train": [row for row in quality_rows if row.get("window_id") in predictor_train_window_ids],
        "calibration": [row for row in quality_rows if row.get("window_id") in calibration_window_ids],
        "evaluation_dev": [row for row in quality_rows if row.get("window_id") in evaluation_window_ids],
    }
    if any(not rows for rows in rows_by_split.values()):
        raise RuntimeError("one or more predictor splits contain no quality rows")

    calibration_labels = [int(float(row["handoff_label"])) for row in rows_by_split["calibration"]]
    calibration_raw_probabilities = [float(row["handoff_probability"]) for row in rows_by_split["calibration"]]
    calibration_logits = [logit(value) for value in calibration_raw_probabilities]
    evaluation_labels = [int(float(row["handoff_label"])) for row in rows_by_split["evaluation_dev"]]
    evaluation_raw_probabilities = [float(row["handoff_probability"]) for row in rows_by_split["evaluation_dev"]]
    evaluation_logits = [logit(value) for value in evaluation_raw_probabilities]

    identity_parameters = identity_calibration("handoff", len(calibration_labels))
    temperature_parameters = fit_binary_temperature(calibration_labels, calibration_logits, seed=args.random_seed)
    calibration_identity_metrics = binary_calibration_metrics(calibration_labels, calibration_raw_probabilities)
    calibration_temperature_probabilities = apply_binary_temperature(
        calibration_logits, temperature_parameters["temperature"]
    )
    calibration_temperature_metrics = binary_calibration_metrics(
        calibration_labels, calibration_temperature_probabilities
    )
    selection = select_calibration_method(
        [
            {
                **identity_parameters,
                "metrics": calibration_identity_metrics["threshold_independent"],
            },
            {
                **temperature_parameters,
                "metrics": calibration_temperature_metrics["threshold_independent"],
            },
        ]
    )
    selected_temperature = float(selection["selected"].get("temperature", 1.0))
    evaluation_calibrated_probabilities = apply_binary_temperature(evaluation_logits, selected_temperature)
    threshold_selection = select_abstention_threshold(
        calibration_labels,
        apply_binary_temperature(calibration_logits, selected_temperature),
        minimum_coverage=args.minimum_coverage,
        candidate_thresholds=[index / 100.0 for index in range(0, 101, 5)],
    )
    if threshold_selection["selected_threshold"] is None:
        raise RuntimeError("calibration-only abstention threshold selection is unavailable")
    selected_threshold = float(threshold_selection["selected_threshold"])
    before_binary = binary_calibration_metrics(
        evaluation_labels, evaluation_raw_probabilities, threshold=0.5
    )
    after_binary = binary_calibration_metrics(
        evaluation_labels, evaluation_calibrated_probabilities, threshold=0.5
    )
    evaluation_risk_coverage = risk_coverage_curve(
        evaluation_labels,
        evaluation_calibrated_probabilities,
        candidate_thresholds=threshold_selection["candidate_grid"],
    )
    selected_eval_row = next(
        row for row in evaluation_risk_coverage if row["confidence_threshold"] == selected_threshold
    )
    class_names = [*read_json(manifest_path)["rsu_label_map"]["rsu_ids"], "__none__"]
    next_hard = hard_classification_metrics(
        rows_by_split["evaluation_dev"],
        label_key="next_rsu_label",
        prediction_key="next_rsu_pred_index",
        class_names=class_names,
        empty_label_class="__none__",
    )
    target_rows = [row for row in rows_by_split["evaluation_dev"] if row.get("handoff_target_label")]
    target_hard = hard_classification_metrics(
        target_rows,
        label_key="handoff_target_label",
        prediction_key="handoff_target_pred_index",
        class_names=class_names,
    )
    eta_rows = [row for row in rows_by_split["evaluation_dev"] if int(float(row["handoff_label"])) == 1]
    eta_metrics = eta_regression_metrics(
        [float(row["eta_label_steps"]) for row in eta_rows],
        [float(row["eta_pred_steps"]) for row in eta_rows],
    )

    calibration_payload = {
        "calibration_artifact_contract_version": CALIBRATION_ARTIFACT_CONTRACT_VERSION,
        "calibration_method_version": TEMPERATURE_SCALING_VERSION,
        "fit_split": "calibration",
        "fit_split_manifest_sha256": overlap["split_manifest_sha256"],
        "evaluation_labels_used_for_fit": False,
        "rl_reward_used_for_selection": False,
        "parameters": {
            "handoff": {
                **selection["selected"],
                "selection_rule": selection["selection_rule"],
            },
            "next_rsu": {
                **identity_calibration("next_rsu", 0),
                "availability": "identity_only_historical_full_logits_unavailable",
            },
            "handoff_target": {
                **identity_calibration("handoff_target", 0),
                "availability": "identity_only_historical_full_logits_unavailable",
            },
        },
        "abstention": {
            **{key: value for key, value in threshold_selection.items() if key != "risk_coverage_rows"},
            "selected_threshold": selected_threshold,
        },
        "seed": args.random_seed,
        "predictor_checkpoint_sha256": sha256_file(checkpoint_path),
        "git_commit_at_generation": git_commit,
    }
    calibration_path = output_dir / "calibration_parameters.json"
    write_json(calibration_path, calibration_payload)
    loaded_calibration = load_calibration_artifact(calibration_path)
    snapshot_examples, snapshot_validation = build_snapshot_examples(loaded_calibration, git_commit)
    real_trace = run_real_trace(
        args=args,
        calibration_path=calibration_path,
        checkpoint_path=checkpoint_path,
        git_commit=git_commit,
    )
    observed_snapshots = list(snapshot_examples)
    observed_snapshots.extend(real_trace.get("snapshot_examples", []))

    split_manifest = {
        **overlap["split_manifest"],
        "manifest_sha256": overlap["split_manifest_sha256"],
        "partition_rule": "train plan order index modulo 4 equals 3 -> calibration; others -> predictor_train",
        "partition_rule_frozen_before_metrics": True,
        "sample_count_by_split": {name: len(rows) for name, rows in rows_by_split.items()},
        "formal_holdout_hidden_used": False,
        "evaluation_role": "existing non-hidden dev only",
    }
    split_overlap = {
        **{key: value for key, value in overlap.items() if key != "split_manifest"},
        "vehicle_group_overlap_audit": audit_vehicle_group_overlap(rows_by_split),
        "interpretation": "vehicle IDs may recur, but raw frame/time intervals are disjoint; interval isolation is authoritative",
    }
    metrics_before = {
        "binary_handoff": before_binary,
        "next_rsu": next_hard,
        "handoff_target_eligible_only": target_hard,
        "eta_handoff_positive_only": eta_metrics,
        "probability_source": "historical v112 quality rows",
    }
    metrics_after = {
        "binary_handoff": after_binary,
        "next_rsu": next_hard,
        "handoff_target_eligible_only": target_hard,
        "eta_handoff_positive_only": eta_metrics,
        "selected_evaluation_risk_coverage": selected_eval_row,
        "calibration_improvement_is_not_assumed": True,
    }
    causality_audit = {
        "reviewed_at": reviewed_at,
        "audit_version": CALIBRATION_AUDIT_VERSION,
        "snapshot_contract_version": CAUSAL_PREDICTOR_SNAPSHOT_CONTRACT_VERSION,
        "source_document": str(DEFAULT_AUDIT_DOC.relative_to(ROOT_DIR)),
        "source_document_sha256": sha256_file(DEFAULT_AUDIT_DOC),
        "runtime_future_label_consumed": False,
        "runtime_reward_or_action_consumed": False,
        "label_horizon_steps": 3,
        "history_reset_at_window_or_episode": True,
        "delayed_snapshot_recomputed": False,
        "oracle_identity_isolated": True,
        "claim_boundary": "contract and nonformal predictor diagnostics only; no policy-benefit or digital-twin claim",
    }
    input_provenance = {
        "quality_rows": {"path": str(quality_path.resolve()), "sha256": sha256_file(quality_path)},
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "predictor_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "train_window_plan": {"path": str(train_plan_path.resolve()), "sha256": sha256_file(train_plan_path)},
        "evaluation_window_plan": {"path": str(evaluation_plan_path.resolve()), "sha256": sha256_file(evaluation_plan_path)},
        "NGSIM_runtime_trace": {
            "availability": real_trace.get("availability"),
            "source_path": real_trace.get("source_path"),
            "payload_sha256_not_computed_reason": "large raw source remains local and is not copied into artifact",
        },
        "G11_cross_source_payload_downloaded": False,
    }

    outputs = {
        "predictor_causality_audit.json": causality_audit,
        "split_manifest.json": split_manifest,
        "split_overlap_audit.json": split_overlap,
        "calibration_config.json": {
            "fit_split": "calibration",
            "evaluation_split": "evaluation_dev",
            "method_candidates": ["identity", "binary_temperature_scaling"],
            "method_selection_rule": selection["selection_rule"],
            "threshold_selection_rule": threshold_selection["selection_rule"],
            "minimum_coverage": args.minimum_coverage,
            "reliability_bin_contract_version": RELIABILITY_BIN_CONTRACT_VERSION,
            "uses_rl_reward": False,
        },
        "metrics_before_calibration.json": metrics_before,
        "metrics_after_calibration.json": metrics_after,
        "reliability_diagram_rows.json": {
            "before": before_binary["reliability"],
            "after": after_binary["reliability"],
        },
        "risk_coverage_rows.json": {
            "calibration": threshold_selection["risk_coverage_rows"],
            "evaluation": evaluation_risk_coverage,
            "selected_threshold": selected_threshold,
            "selected_evaluation_row": selected_eval_row,
        },
        "staleness_audit.json": staleness_diagnostics(observed_snapshots, update_intervals=(1, 3, 6, 12)),
        "snapshot_examples.json": {
            "synthetic_contract_examples": snapshot_examples,
            "real_NGSIM_minimal_examples": real_trace.get("snapshot_examples", []),
        },
        "snapshot_validation.json": {
            **snapshot_validation,
            "real_NGSIM_minimal": {
                "availability": real_trace.get("availability"),
                "snapshot_count": real_trace.get("snapshot_count"),
                "accepted_snapshot_count": real_trace.get("accepted_snapshot_count"),
                "abstention_reason_counts": real_trace.get("abstention_reason_counts"),
            },
        },
        "decision_observation_trace_sample.json": real_trace,
        "input_provenance.json": input_provenance,
        "command_log.json": {
            "reviewed_at": reviewed_at,
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "git_commit": git_commit,
            "formal_holdout_hidden_executed": False,
            "training_or_tuning_executed": False,
            "G13_executed": False,
        },
    }
    for name, payload in outputs.items():
        write_json(output_dir / name, payload)

    integrity_rows = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "artifact_integrity_manifest.json":
            continue
        integrity_rows.append(
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    integrity = {
        "artifact_integrity_contract_version": "g12_artifact_integrity_v1.0.0",
        "run_id": args.run_id,
        "file_count_excluding_manifest": len(integrity_rows),
        "files": integrity_rows,
        "aggregate_sha256": canonical_sha256(integrity_rows),
    }
    write_json(output_dir / "artifact_integrity_manifest.json", integrity)
    print(json.dumps({
        "output_dir": str(output_dir),
        "split_manifest_sha256": overlap["split_manifest_sha256"],
        "selected_temperature": selected_temperature,
        "selected_threshold": selected_threshold,
        "evaluation_before": before_binary["threshold_independent"],
        "evaluation_after": after_binary["threshold_independent"],
        "real_trace_availability": real_trace.get("availability"),
        "artifact_aggregate_sha256": integrity["aggregate_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
