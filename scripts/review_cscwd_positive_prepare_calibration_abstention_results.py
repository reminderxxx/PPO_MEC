"""Create a bounded read-only review of the completed calibration A/B."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_20261011_v1"
RUN_ROOT = ROOT / "artifacts/experiments" / RUN_ID
ANALYSIS_ROOT = ROOT / "artifacts/analysis" / f"{RUN_ID}_analysis_v1"
SUPERVISOR_ROOT = ROOT / "artifacts/experiments" / f"{RUN_ID}_supervisor"
OUTPUT_ROOT = ROOT / "artifacts/analysis" / f"{RUN_ID}_result_review_v1"
EXPECTED_COMMIT = "56d25519a6572fe6822cd5a1c99dcf1a6346929b"
ARMS = ("sa_public_v3", "sa_calibration_abstention", "ppo", "mappo")
VIEWS = ("selected", "fixed_update24")
SEEDS = (7, 17, 29, 43, 61)
METRICS = (
    "reward", "reward_node_completion", "reward_workflow_completion",
    "reward_time_penalty", "reward_transfer_penalty", "reward_failure_penalty",
    "reward_deadline_penalty", "workflow_completion_rate",
    "on_time_workflow_completion_rate", "service_failure_rate", "service_failures",
    "unfinished_rate", "truncated", "elapsed_seconds", "total_transfer_bytes",
    "input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
    "backhaul_transfer_bytes", "partial_transfer_bytes", "recompute_seconds",
    "migration_successes", "target_departed_before_delivery",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_integrity(
    root: Path, *, require_manifest: bool = True,
) -> dict[str, Any]:
    manifest_path = root / "artifact_integrity.json"
    if not manifest_path.is_file():
        if require_manifest:
            raise FileNotFoundError(manifest_path)
        files = {
            str(path.relative_to(root)): _sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()
        }
        return {
            "source_manifest_present": False,
            "manifest_sha256": None,
            "listed_file_count": 0,
            "independently_hashed_file_count": len(files),
            "independently_computed_files": files,
            "mismatch_count": 0,
            "mismatches": [],
            "limitation": (
                "source root has no artifact_integrity.json; files were hashed "
                "read-only but cannot be matched to a producer manifest"
            ),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for row in manifest["files"]:
        path = root / row["path"]
        actual = _sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({
                "path": row["path"], "expected": row["sha256"], "actual": actual,
            })
    return {
        "manifest_sha256": _sha256(manifest_path),
        "listed_file_count": len(manifest["files"]),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _number(value: Any) -> float:
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text in {"false", ""}:
        return 0.0
    return float(text)


def _network_hash(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name, tensor in checkpoint["network_state_dict"].items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _augment_evaluations(
    evaluations: list[dict[str, str]], behavior: list[dict[str, str]],
) -> list[dict[str, Any]]:
    fields = (
        "arm", "view", "seed", "time_profile", "condition_id",
        "source_segment_id", "window_id",
    )
    behavior_by_key: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in behavior:
        behavior_by_key[tuple(row[field] for field in fields)].append(row)
    output = []
    for source in evaluations:
        row: dict[str, Any] = dict(source)
        row["elapsed_seconds"] = _number(
            source["completed_sample_elapsed_seconds"]
            or source["elapsed_seconds_for_failed_or_truncated"]
        )
        row["total_transfer_bytes"] = round(
            _number(source["total_transfer_mb"]) * 1_000_000.0
        )
        ledger = behavior_by_key[tuple(source[field] for field in fields)]
        row["target_departed_before_delivery"] = sum(
            step.get("failure_reason") == "target_departed_before_delivery"
            for step in ledger
        )
        output.append(row)
    return output


def _aggregate(
    rows: list[dict[str, Any]], group_fields: Iterable[str],
) -> list[dict[str, Any]]:
    group_fields = tuple(group_fields)
    grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output = []
    for key in sorted(grouped):
        subset = grouped[key]
        result: dict[str, Any] = {
            field: value for field, value in zip(group_fields, key)
        }
        result["episode_count"] = len(subset)
        result.update({
            metric: float(np.mean([_number(row[metric]) for row in subset]))
            for metric in METRICS
        })
        output.append(result)
    return output


def _full_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = (
        "arm", "method", "view", "seed", "time_profile", "condition_id",
        "source_segment_id", "window_id",
    )
    return [
        {
            **{field: row.get(field, "") for field in identity},
            **{metric: _number(row[metric]) for metric in METRICS},
        }
        for row in rows
    ]


def _paired_sa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = (
        "view", "seed", "time_profile", "condition_id", "source_segment_id",
        "window_id",
    )
    control = {
        tuple(str(row[field]) for field in key_fields): row
        for row in rows if row["arm"] == "sa_public_v3"
    }
    candidate = {
        tuple(str(row[field]) for field in key_fields): row
        for row in rows if row["arm"] == "sa_calibration_abstention"
    }
    if control.keys() != candidate.keys() or len(control) != 80:
        raise RuntimeError("SA pair identity drift")
    output = []
    for key in sorted(control):
        result: dict[str, Any] = {
            field: value for field, value in zip(key_fields, key)
        }
        result.update({
            f"{metric}_delta": (
                _number(candidate[key][metric]) - _number(control[key][metric])
            )
            for metric in METRICS
        })
        output.append(result)
    return output


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"create-only output exists: {OUTPUT_ROOT}")
    terminal = json.loads((SUPERVISOR_ROOT / "terminal_receipt.json").read_text())
    completion = json.loads((RUN_ROOT / "completion_receipt.json").read_text())
    manifest = json.loads((RUN_ROOT / "run_manifest.json").read_text())
    original_analysis = json.loads(
        (ANALYSIS_ROOT / "analysis_summary.json").read_text()
    )
    if not (
        terminal.get("status") == "PASS"
        and terminal.get("classification") == "SCIENTIFIC_CHILD_COMPLETE"
        and terminal.get("return_code") == 0
        and completion.get("scientific_execution_complete") is True
        and manifest.get("git_commit") == EXPECTED_COMMIT
        and completion.get("git_commit") == EXPECTED_COMMIT
    ):
        raise RuntimeError("terminal or scientific identity is incomplete")
    expected_counts = {
        "learned_cells": 20, "training_steps": 23040,
        "optimizer_steps": 3840, "checkpoint_selection_episodes": 80,
        "final_learned_evaluation_episodes": 320,
    }
    if any(manifest["counts"].get(key) != value for key, value in expected_counts.items()):
        raise RuntimeError("scientific budget receipt drift")

    run_integrity = _verify_integrity(RUN_ROOT)
    analysis_integrity = _verify_integrity(ANALYSIS_ROOT)
    supervisor_integrity = _verify_integrity(
        SUPERVISOR_ROOT, require_manifest=False
    )
    if any(
        row["mismatch_count"]
        for row in (run_integrity, analysis_integrity, supervisor_integrity)
    ):
        raise RuntimeError("artifact integrity mismatch")

    evaluations = _read_csv(RUN_ROOT / "evaluation_rows.csv")
    behavior = _read_csv(RUN_ROOT / "behavior_rows.csv")
    signals = _read_csv(RUN_ROOT / "training_signal_rows.csv")
    rules = _read_csv(RUN_ROOT / "rule_evaluation_rows.csv")
    rule_behavior = _read_csv(RUN_ROOT / "rule_behavior_rows.csv")
    summaries = json.loads((RUN_ROOT / "training_summary.json").read_text())
    if len(evaluations) != 320 or len(signals) != 23040 or len(rules) != 18:
        raise RuntimeError("result row count drift")
    rows = _augment_evaluations(evaluations, behavior)
    full_rows = _full_rows(rows)
    paired = _paired_sa(rows)

    labels: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in signals:
        labels[row["arm"]][
            (row["event_label"], row["event_supervision_weight"])
        ] += 1
    expected_all_abstain = {
        arm: Counter({("abstain", "0.0"): 5760}) for arm in ARMS
    }
    if dict(labels) != expected_all_abstain:
        raise RuntimeError("unexpected training label exposure")

    summary_by_arm_seed = {
        (row["arm"], int(row["seed"])): row for row in summaries
    }
    checkpoint_identity = []
    for seed in SEEDS:
        for view, suffix in (("selected", "selected"), ("fixed_update24", "update24")):
            control_path = RUN_ROOT / "checkpoints" / f"sa_public_v3_seed{seed}_{suffix}.pt"
            candidate_path = RUN_ROOT / "checkpoints" / f"sa_calibration_abstention_seed{seed}_{suffix}.pt"
            control_hash = _network_hash(control_path)
            candidate_hash = _network_hash(candidate_path)
            checkpoint_identity.append({
                "seed": seed, "view": view,
                "control_network_tensor_sha256": control_hash,
                "candidate_network_tensor_sha256": candidate_hash,
                "network_tensors_equal": control_hash == candidate_hash,
                "selected_update_control": summary_by_arm_seed[("sa_public_v3", seed)]["selected_update"],
                "selected_update_candidate": summary_by_arm_seed[("sa_calibration_abstention", seed)]["selected_update"],
            })
    all_networks_equal = all(row["network_tensors_equal"] for row in checkpoint_identity)
    all_eval_deltas_zero = all(
        all(abs(float(row[f"{metric}_delta"])) <= 1e-12 for metric in METRICS)
        for row in paired
    )
    if not all_networks_equal or not all_eval_deltas_zero:
        raise RuntimeError("zero-exposure SA identity expectation failed")

    method_view_profile = _aggregate(
        rows, ("arm", "view", "time_profile")
    )
    seed_view_profile = _aggregate(
        rows, ("arm", "view", "seed", "time_profile")
    )
    source_condition = _aggregate(
        rows,
        ("arm", "view", "time_profile", "condition_id", "source_segment_id"),
    )
    rule_augmented = _augment_evaluations(rules, rule_behavior)
    rule_rows = _full_rows(rule_augmented)
    for output, source in zip(rule_rows, rules):
        output.update({
            field: source.get(field, "")
            for field in (
                "capability_profile", "phase_v2_independent_baseline",
                "phase_v2_available",
            )
        })
    rule_profile_summary = _aggregate(
        rule_augmented, ("method", "view", "time_profile")
    )

    update_rows = _read_csv(RUN_ROOT / "training_updates.csv")
    update_supervision = {}
    action_counts = {}
    for arm in ARMS:
        arm_updates = [row for row in update_rows if row["arm"] == arm]
        arm_signals = [row for row in signals if row["arm"] == arm]
        action_counts[arm] = dict(sorted(Counter(
            row["executed_action"] for row in arm_signals
        ).items()))
        if arm.startswith("sa_"):
            update_supervision[arm] = {
                field: sum(int(float(row[field])) for row in arm_updates)
                for field in (
                    "event_aux_supervision_eligible_count",
                    "event_aux_supervision_supervised_count",
                    "event_aux_supervision_abstained_count",
                )
            }

    review = {
        "schema_version": "cscwd_positive_prepare_calibration_abstention_result_review_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "literature_cutoff": "2026-10-11",
        "target_venue": "IEEE TMC",
        "artifact_run_id": RUN_ID,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": EXPECTED_COMMIT,
        "evidence_level": "E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY",
        "scientific_execution": "PASS",
        "original_analysis_verdict_preserved": original_analysis["verdict"],
        "causal_candidate_verdict": "UNINFORMATIVE_ZERO_TREATMENT_EXPOSURE",
        "reason": (
            "both SA arms observed zero supervised positive-prepare training rows; "
            "all ten selected/fixed network tensor pairs and all 80 evaluation pairs match"
        ),
        "integrity": {
            "run": run_integrity,
            "analysis": analysis_integrity,
            "supervisor": supervisor_integrity,
        },
        "counts": {
            **expected_counts,
            "checkpoint_files": len(list((RUN_ROOT / "checkpoints").glob("*.pt"))),
            "learned_evaluation_rows": len(evaluations),
            "rule_evaluation_rows": len(rules),
            "paired_sa_rows": len(paired),
            "formal_or_holdout_reads": 0,
            "automatic_retry": False,
        },
        "training_label_counts": {
            arm: {
                f"{label}|weight={weight}": count
                for (label, weight), count in sorted(labels[arm].items())
            }
            for arm in ARMS
        },
        "training_action_counts": action_counts,
        "sa_update_supervision_counts": update_supervision,
        "label_path_diagnosis": {
            "training_source_segment": "lankershim",
            "checkpoint_selection_source_segment": "us_101",
            "static_prepare_witness_source": "us_101",
            "consumer_switch_present": True,
            "control_and_candidate_base_label_exposure": "all_abstain",
            "specific_prerequisite_failure_recoverable": False,
            "unrecoverable_because": (
                "training_signal_rows.csv did not persist semantic_state or the "
                "public label reason"
            ),
            "located_protocol_defect": (
                "authorization lacked a nonzero training-source base-positive "
                "label-exposure gate"
            ),
        },
        "sa_identity": {
            "checkpoint_pair_count": len(checkpoint_identity),
            "all_network_tensor_pairs_equal": all_networks_equal,
            "all_evaluation_metric_deltas_zero": all_eval_deltas_zero,
            "checkpoint_pairs": checkpoint_identity,
        },
        "four_unverified": [
            {
                "item": "real_cross_rsu_net_benefit",
                "status": "UNVERIFIED",
                "missing": (
                    "deployed multi-RSU measurements for radio/backhaul/compute, "
                    "state serialization, failures, latency, and bytes"
                ),
            },
            {
                "item": "sharing_by_migration_causal_interaction",
                "status": "UNVERIFIED",
                "missing": (
                    "pre-registered 2x2 sharing on/off by migration on/off factorial "
                    "with difference-in-differences on service and resource metrics"
                ),
            },
            {
                "item": "three_head_structure_beats_fair_baselines",
                "status": "UNVERIFIED",
                "missing": (
                    "parameter-matched full three-head versus shared/single-head "
                    "structural ablation under identical observations, PPO/aux budget, "
                    "seeds, selection, and retained PPO/MAPPO/public-rule baselines"
                ),
            },
            {
                "item": "independent_generalization_paper_ready",
                "status": "UNVERIFIED",
                "missing": (
                    "source-disjoint untouched mobility units; current Lankershim and "
                    "US-101 are exposed and new seeds are not independent units"
                ),
            },
        ],
        "claim_boundary": (
            "the run is a valid completed experiment but contains no active auxiliary "
            "treatment; it cannot support calibration-abstention efficacy, three-head "
            "innovation, real-RSU benefit, interaction, generalization, or paper readiness"
        ),
    }

    OUTPUT_ROOT.mkdir(parents=True)
    _write_json(OUTPUT_ROOT / "review.json", review)
    _write_csv(OUTPUT_ROOT / "learned_full_strata.csv", full_rows)
    _write_csv(OUTPUT_ROOT / "rule_full_strata.csv", rule_rows)
    _write_csv(OUTPUT_ROOT / "method_view_profile_summary.csv", method_view_profile)
    _write_csv(OUTPUT_ROOT / "seed_view_profile_summary.csv", seed_view_profile)
    _write_csv(OUTPUT_ROOT / "source_condition_summary.csv", source_condition)
    _write_csv(OUTPUT_ROOT / "rule_profile_summary.csv", rule_profile_summary)
    _write_csv(OUTPUT_ROOT / "paired_sa_deltas.csv", paired)
    _write_csv(OUTPUT_ROOT / "sa_checkpoint_tensor_identity.csv", checkpoint_identity)
    report = """# Calibration-abstention A/B result review

The scientific run is complete and internally consistent, but the causal A/B is
uninformative: all four learned arms logged only `abstain, weight=0`; both SA arms
therefore had zero supervised positive-prepare rows. The ten selected/fixed SA
network tensor pairs and all 80 paired evaluation metric vectors are identical.

The original `MIXED` analysis is preserved. It must not be reinterpreted as candidate
support or falsification. Full learned rows, rule rows, every seed/view/profile, and
source/condition summaries are stored alongside this review. The four requested
innovation/generalization claims remain UNVERIFIED for the reasons in `review.json`.
"""
    (OUTPUT_ROOT / "report.md").write_text(report, encoding="utf-8")
    files = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if path.name == "artifact_integrity.json":
            continue
        files.append({
            "path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size,
        })
    _write_json(OUTPUT_ROOT / "artifact_integrity.json", {
        "schema_version": "cscwd_result_review_integrity_v1",
        "files": files,
    })
    print(json.dumps({
        "status": "PASS",
        "output_root": str(OUTPUT_ROOT),
        "causal_candidate_verdict": review["causal_candidate_verdict"],
        "all_network_tensor_pairs_equal": all_networks_equal,
        "all_evaluation_metric_deltas_zero": all_eval_deltas_zero,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
