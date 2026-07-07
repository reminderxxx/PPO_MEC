"""Train a thin supervised short-horizon handoff predictor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    build_selected_workflow_states,
    clone_rsu_state,
    load_window_bundle,
)
from src.predictors import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    SupervisedHandoffPredictorNetwork,
    build_feature_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised short-horizon handoff predictor")
    parser.add_argument(
        "--train_window_plan_path",
        type=str,
        default=str(ROOT_DIR / "configs" / "experiment" / "top_journal_v8_strict_split_20260621" / "train_window_plan.json"),
    )
    parser.add_argument(
        "--dev_window_plan_path",
        type=str,
        default=str(ROOT_DIR / "configs" / "experiment" / "top_journal_v8_strict_split_20260621" / "dev_window_plan.json"),
    )
    parser.add_argument("--mobility_source", type=str, default="ngsim", choices=["ngsim", "lust"])
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--max_mobility_rows", type=int, default=10000)
    parser.add_argument("--max_workflows", type=int, default=1)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "training" / "supervised_predictors"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def load_window_plan(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = Path(path)
    payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    plan = payload.get("selected_window_plan", payload if isinstance(payload, list) else [])
    if not isinstance(plan, list) or not plan:
        raise ValueError(f"selected_window_plan missing or empty: {plan_path}")
    return payload if isinstance(payload, dict) else {}, [dict(item) for item in plan]


def label_index(rsu_id: str | None, rsu_ids: list[str]) -> int:
    if rsu_id is None:
        return len(rsu_ids)
    try:
        return rsu_ids.index(str(rsu_id))
    except ValueError:
        return len(rsu_ids)


def first_handoff_target(
    *,
    current_rsu_id: str | None,
    sequence: list[str | None],
) -> tuple[str | None, int]:
    for index, rsu_id in enumerate(sequence, start=1):
        if rsu_id is not None and str(rsu_id) != str(current_rsu_id):
            return str(rsu_id), index
    return None, 0


def build_samples_for_plan(
    *,
    plan: list[dict[str, Any]],
    args: argparse.Namespace,
    workflow_state: Any,
    rsu_ids: list[str] | None,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    resolved_rsu_ids = list(rsu_ids or [])
    for window in plan:
        bundle = load_window_bundle(
            root_dir=ROOT_DIR,
            mobility_source=args.mobility_source,
            mobility_csv_path=args.mobility_csv_path,
            lust_scenario_root=args.lust_scenario_root,
            max_mobility_rows=args.max_mobility_rows,
            rsu_layout=str(window.get("recommended_rsu_layout", args.rsu_layout)),
            frame_offset=int(window["frame_offset"]),
            window_length=int(window["window_length"]),
            random_seed=args.random_seed,
        )
        window_rsu_ids = [str(rsu.rsu_id) for rsu in bundle.rsu_states]
        if not resolved_rsu_ids:
            resolved_rsu_ids = window_rsu_ids
        if window_rsu_ids != resolved_rsu_ids:
            raise ValueError(
                f"RSU layout mismatch in {window.get('window_id')}: {window_rsu_ids} != {resolved_rsu_ids}"
            )
        mapper = RSUMapper([clone_rsu_state(rsu_state) for rsu_state in bundle.rsu_states])
        last_positions: dict[str, tuple[float, float]] = {}
        frames = list(bundle.frames)
        for frame_index, frame in enumerate(frames):
            vehicles = list(frame.get("vehicles", []))
            current_associations = mapper.associate(vehicles)
            for vehicle in vehicles:
                future_sequence: list[str | None] = []
                for offset in range(1, int(args.horizon) + 1):
                    future_index = frame_index + offset
                    if future_index >= len(frames):
                        future_sequence.append(None)
                        continue
                    future_vehicles = list(frames[future_index].get("vehicles", []))
                    future_associations = mapper.associate(future_vehicles)
                    future_sequence.append(future_associations.get(vehicle.vehicle_id))
                current_rsu_id = current_associations.get(vehicle.vehicle_id)
                target_rsu_id, eta_steps = first_handoff_target(
                    current_rsu_id=current_rsu_id,
                    sequence=future_sequence,
                )
                feature_vector = build_feature_vector(
                    vehicle=vehicle,
                    rsu_states=bundle.rsu_states,
                    workflow_state=workflow_state,
                    current_associations=current_associations,
                    rsu_ids=resolved_rsu_ids,
                    last_vehicle_positions=last_positions,
                )
                rows.append(
                    {
                        "split": split_name,
                        "window_id": str(window.get("window_id", bundle.rsu_metadata.get("window_id", ""))),
                        "window_class": str(window.get("window_class", bundle.rsu_metadata.get("window_class", ""))),
                        "time_index": int(frame.get("time_index", frame_index)),
                        "vehicle_id": str(vehicle.vehicle_id),
                        "current_rsu_id": current_rsu_id,
                        "next_rsu_label": future_sequence[0] if future_sequence else None,
                        "handoff_target_label": target_rsu_id,
                        "handoff_within_horizon": 1.0 if target_rsu_id is not None else 0.0,
                        "handoff_eta_steps": float(eta_steps if eta_steps > 0 else int(args.horizon) + 1),
                        "features": feature_vector,
                    }
                )
            for vehicle in vehicles:
                last_positions[vehicle.vehicle_id] = (float(vehicle.position_x), float(vehicle.position_y))
            if args.sample_limit and len(rows) >= int(args.sample_limit):
                return rows[: int(args.sample_limit)], resolved_rsu_ids
    return rows, resolved_rsu_ids


def tensors_from_rows(rows: list[dict[str, Any]], rsu_ids: list[str]) -> tuple[torch.Tensor, ...]:
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    next_labels = torch.tensor([label_index(row["next_rsu_label"], rsu_ids) for row in rows], dtype=torch.long)
    target_labels = torch.tensor([label_index(row["handoff_target_label"], rsu_ids) for row in rows], dtype=torch.long)
    handoff_labels = torch.tensor([float(row["handoff_within_horizon"]) for row in rows], dtype=torch.float32)
    eta_labels = torch.tensor([float(row["handoff_eta_steps"]) for row in rows], dtype=torch.float32)
    return features, next_labels, target_labels, handoff_labels, eta_labels


def binary_auc(labels: list[float], scores: list[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels) if label > 0.5]
    negatives = [(score, label) for score, label in zip(scores, labels) if label <= 0.5]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive_score, _ in positives:
        for negative_score, _ in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / float(len(positives) * len(negatives))


def expected_calibration_error(labels: list[float], scores: list[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            (label, score)
            for label, score in zip(labels, scores)
            if (score >= lower and (score < upper or bin_index == bins - 1))
        ]
        if not members:
            continue
        mean_label = sum(label for label, _ in members) / len(members)
        mean_score = sum(score for _, score in members) / len(members)
        ece += (len(members) / total) * abs(mean_label - mean_score)
    return ece


def evaluate_model(
    *,
    model: SupervisedHandoffPredictorNetwork,
    rows: list[dict[str, Any]],
    rsu_ids: list[str],
    handoff_threshold: float = 0.5,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not rows:
        return {}, []
    features, next_labels, target_labels, handoff_labels, eta_labels = tensors_from_rows(rows, rsu_ids)
    with torch.no_grad():
        output = model(features)
        next_probs = torch.softmax(output["next_rsu_logits"], dim=-1)
        target_probs = torch.softmax(output["handoff_target_logits"], dim=-1)
        handoff_probs = torch.sigmoid(output["handoff_logit"])
        eta_pred = output["eta_steps"]
    next_pred = torch.argmax(next_probs, dim=-1)
    target_pred = torch.argmax(target_probs, dim=-1)
    calibrated_threshold = max(0.0, min(float(handoff_threshold), 1.0))
    handoff_pred = (handoff_probs >= calibrated_threshold).float()
    next_accuracy = float((next_pred == next_labels).float().mean().item())
    target_accuracy = float((target_pred == target_labels).float().mean().item())
    true_positive = float(((handoff_pred == 1.0) & (handoff_labels == 1.0)).float().sum().item())
    false_positive = float(((handoff_pred == 1.0) & (handoff_labels == 0.0)).float().sum().item())
    false_negative = float(((handoff_pred == 0.0) & (handoff_labels == 1.0)).float().sum().item())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-9)
    labels_list = [float(item) for item in handoff_labels.tolist()]
    scores_list = [float(item) for item in handoff_probs.tolist()]
    brier = sum((score - label) ** 2 for score, label in zip(scores_list, labels_list)) / max(len(labels_list), 1)
    positive_eta_errors = [
        abs(float(pred) - float(label))
        for pred, label, handoff in zip(eta_pred.tolist(), eta_labels.tolist(), handoff_labels.tolist())
        if float(handoff) > 0.5
    ]
    eta_mae = sum(positive_eta_errors) / max(len(positive_eta_errors), 1)
    metrics = {
        "sample_count": float(len(rows)),
        "handoff_threshold": round(calibrated_threshold, 6),
        "next_rsu_accuracy": round(next_accuracy, 6),
        "handoff_target_accuracy": round(target_accuracy, 6),
        "handoff_precision": round(precision, 6),
        "handoff_recall": round(recall, 6),
        "handoff_f1": round(f1, 6),
        "handoff_auc": round(binary_auc(labels_list, scores_list), 6),
        "handoff_brier_score": round(brier, 6),
        "handoff_ece": round(expected_calibration_error(labels_list, scores_list), 6),
        "handoff_eta_mae": round(eta_mae, 6),
    }
    quality_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        quality_rows.append(
            {
                key: row.get(key)
                for key in [
                    "split",
                    "window_id",
                    "window_class",
                    "time_index",
                    "vehicle_id",
                    "current_rsu_id",
                    "next_rsu_label",
                    "handoff_target_label",
                ]
            }
        )
        quality_rows[-1].update(
            {
                "handoff_label": float(handoff_labels[index].item()),
                "next_rsu_pred_index": int(next_pred[index].item()),
                "handoff_target_pred_index": int(target_pred[index].item()),
                "handoff_probability": round(float(handoff_probs[index].item()), 6),
                "handoff_predicted": int(handoff_pred[index].item()),
                "eta_pred_steps": round(float(eta_pred[index].item()), 6),
                "eta_label_steps": round(float(eta_labels[index].item()), 6),
            }
        )
    return metrics, quality_rows


def select_handoff_threshold(
    *,
    model: SupervisedHandoffPredictorNetwork,
    rows: list[dict[str, Any]],
    rsu_ids: list[str],
) -> dict[str, float]:
    """Select the dev-set threshold that maximizes handoff F1."""
    if not rows:
        return {"threshold": 0.5, "handoff_f1": 0.0, "handoff_precision": 0.0, "handoff_recall": 0.0}
    features, _, _, handoff_labels, _ = tensors_from_rows(rows, rsu_ids)
    with torch.no_grad():
        scores = torch.sigmoid(model(features)["handoff_logit"]).tolist()
    candidates = sorted({0.0, 0.5, 1.0, *[float(score) for score in scores]})
    best = {
        "threshold": 0.5,
        "handoff_f1": -1.0,
        "handoff_precision": 0.0,
        "handoff_recall": 0.0,
    }
    labels = [float(label) for label in handoff_labels.tolist()]
    for threshold in candidates:
        predictions = [1.0 if float(score) >= float(threshold) else 0.0 for score in scores]
        true_positive = sum(1.0 for pred, label in zip(predictions, labels) if pred == 1.0 and label == 1.0)
        false_positive = sum(1.0 for pred, label in zip(predictions, labels) if pred == 1.0 and label == 0.0)
        false_negative = sum(1.0 for pred, label in zip(predictions, labels) if pred == 0.0 and label == 1.0)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-9)
        if f1 > float(best["handoff_f1"]) or (
            math.isclose(f1, float(best["handoff_f1"])) and abs(float(threshold) - 0.5) < abs(float(best["threshold"]) - 0.5)
        ):
            best = {
                "threshold": float(threshold),
                "handoff_f1": float(f1),
                "handoff_precision": float(precision),
                "handoff_recall": float(recall),
            }
    return {key: round(value, 6) for key, value in best.items()}


def grouped_metrics(
    *,
    model: SupervisedHandoffPredictorNetwork,
    rows: list[dict[str, Any]],
    rsu_ids: list[str],
    handoff_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("window_class", "unknown")), []).append(row)
    return {
        group_name: evaluate_model(
            model=model,
            rows=group_rows,
            rsu_ids=rsu_ids,
            handoff_threshold=handoff_threshold,
        )[0]
        for group_name, group_rows in sorted(groups.items())
    }


def write_quality_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.random_seed))
    train_metadata, train_plan = load_window_plan(args.train_window_plan_path)
    dev_metadata, dev_plan = load_window_plan(args.dev_window_plan_path)
    workflow_states = build_selected_workflow_states(
        workflow_csv_path=args.workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )
    workflow_state = workflow_states[0]
    AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json")
    train_rows, rsu_ids = build_samples_for_plan(
        plan=train_plan,
        args=args,
        workflow_state=workflow_state,
        rsu_ids=None,
        split_name="train",
    )
    dev_rows, rsu_ids = build_samples_for_plan(
        plan=dev_plan,
        args=args,
        workflow_state=workflow_state,
        rsu_ids=rsu_ids,
        split_name="dev",
    )
    if not train_rows or not dev_rows:
        raise RuntimeError("supervised predictor requires non-empty train and dev samples")
    train_tensors = tensors_from_rows(train_rows, rsu_ids)
    train_dataset = TensorDataset(*train_tensors)
    train_loader = DataLoader(train_dataset, batch_size=max(int(args.batch_size), 1), shuffle=True)
    input_dim = train_tensors[0].shape[1]
    model = SupervisedHandoffPredictorNetwork(
        input_dim=int(input_dim),
        rsu_class_count=len(rsu_ids) + 1,
        hidden_dim=int(args.hidden_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    huber_loss = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for features, next_labels, target_labels, handoff_labels, eta_labels in train_loader:
            optimizer.zero_grad()
            output = model(features)
            positive_mask = handoff_labels > 0.5
            eta_loss = (
                huber_loss(output["eta_steps"][positive_mask], eta_labels[positive_mask])
                if bool(positive_mask.any())
                else output["eta_steps"].sum() * 0.0
            )
            loss = (
                ce_loss(output["next_rsu_logits"], next_labels)
                + ce_loss(output["handoff_target_logits"], target_labels)
                + bce_loss(output["handoff_logit"], handoff_labels)
                + 0.25 * eta_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        model.eval()
        dev_metrics, _ = evaluate_model(model=model, rows=dev_rows, rsu_ids=rsu_ids)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": round(total_loss / max(batch_count, 1), 6),
                "dev_handoff_f1": float(dev_metrics.get("handoff_f1", 0.0)),
                "dev_next_rsu_accuracy": float(dev_metrics.get("next_rsu_accuracy", 0.0)),
            }
        )
    model.eval()
    threshold_selection = select_handoff_threshold(model=model, rows=dev_rows, rsu_ids=rsu_ids)
    handoff_threshold = float(threshold_selection["threshold"])
    train_metrics, train_quality_rows = evaluate_model(
        model=model,
        rows=train_rows,
        rsu_ids=rsu_ids,
        handoff_threshold=handoff_threshold,
    )
    dev_metrics, dev_quality_rows = evaluate_model(
        model=model,
        rows=dev_rows,
        rsu_ids=rsu_ids,
        handoff_threshold=handoff_threshold,
    )
    dev_uncalibrated_metrics, _ = evaluate_model(model=model, rows=dev_rows, rsu_ids=rsu_ids)
    run_id = datetime.now().strftime("supervised_handoff_predictor_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    feature_names = [f"feature_{index}" for index in range(int(input_dim))]
    checkpoint_path = output_root / "supervised_handoff_predictor.pt"
    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit(),
        "horizon": int(args.horizon),
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "feature_schema": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": feature_names,
            "note": "mobility/RSU/workflow/cache state only; no reward/action/outcome features",
        },
        "rsu_label_map": {
            "rsu_ids": list(rsu_ids),
            "none_index": len(rsu_ids),
        },
        "calibration": {
            "handoff_decision_threshold": handoff_threshold,
            "threshold_selection_split": "dev",
            "threshold_selection_metric": "handoff_f1",
            "threshold_selection": threshold_selection,
            "uncalibrated_threshold": 0.5,
        },
        "model_state_dict": model.state_dict(),
        "metrics": {
            "train": train_metrics,
            "dev": dev_metrics,
            "dev_uncalibrated": dev_uncalibrated_metrics,
            "dev_by_window_class": grouped_metrics(
                model=model,
                rows=dev_rows,
                rsu_ids=rsu_ids,
                handoff_threshold=handoff_threshold,
            ),
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    quality_rows_path = output_root / "predictor_quality_rows.csv"
    write_quality_rows(quality_rows_path, train_quality_rows + dev_quality_rows)
    metrics_manifest = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "claim_boundary": "supervised short-horizon handoff predictor; not full digital twin or trajectory SOTA",
        "training_data_policy": "mobility future labels only; no reward/action/checkpoint outcome consumed",
        "git_commit": checkpoint_payload["git_commit"],
        "argv": sys.argv,
        "train_window_plan_path": str(Path(args.train_window_plan_path).resolve()),
        "train_window_plan_sha256": sha256_file(Path(args.train_window_plan_path)),
        "train_window_plan_split": str(train_metadata.get("split", "unknown")),
        "dev_window_plan_path": str(Path(args.dev_window_plan_path).resolve()),
        "dev_window_plan_sha256": sha256_file(Path(args.dev_window_plan_path)),
        "dev_window_plan_split": str(dev_metadata.get("split", "unknown")),
        "horizon": int(args.horizon),
        "calibration": checkpoint_payload["calibration"],
        "rsu_label_map": checkpoint_payload["rsu_label_map"],
        "sample_counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
        },
        "history": history,
        "metrics": checkpoint_payload["metrics"],
        "quality_rows_path": str(quality_rows_path),
    }
    metrics_path = output_root / "predictor_metrics_manifest.json"
    metrics_path.write_text(json.dumps(metrics_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("supervised handoff predictor training complete")
    print(f"run_id: {run_id}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"metrics_manifest_path: {metrics_path}")
    print(f"quality_rows_path: {quality_rows_path}")


if __name__ == "__main__":
    main()
