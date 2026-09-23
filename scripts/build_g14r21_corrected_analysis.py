#!/usr/bin/env python3
"""Build the immutable G14R21 correction and unsigned holdout review package.

This consumer reads frozen formal results and pre-open holdout metadata only.  It
does not load a policy, dispatch an episode, select a window, or open holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manage_typed_model_cache_formal_artifacts import _claim_evidence_rows
from src.runtime.formal_protocol_capabilities import get_protocol_capabilities


PRIMARY_METRICS = (
    "full_service_ready_byte_hit_rate",
    "joint_base_adapter_hit_rate",
    "full_service_ready_request_rate",
    "transfer_mb_per_request",
    "workflow_continuity_rate",
    "end_to_end_workflow_delay",
)
SPLITS = ("train", "dev", "formal", "sealed_holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--corrected-statistics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--g14a01-review", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["baseline_agent"]), str(row["metric"])


def build_claim_and_diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    old_claims = {row_identity(row): row for row in _claim_evidence_rows(old)}
    new_claims = {row_identity(row): row for row in _claim_evidence_rows(new)}
    old_rows = {row_identity(row): row for row in old["rows"]}
    new_rows = {row_identity(row): row for row in new["rows"]}
    if set(old_rows) != set(new_rows) or len(new_rows) != 84:
        raise ValueError("the preregistered 84-comparison family is incomplete")
    differences = []
    for identity in sorted(new_rows):
        before, after = old_rows[identity], new_rows[identity]
        differences.append(
            {
                "baseline_agent": identity[0],
                "metric": identity[1],
                "old_row_level_wins_ties_losses": [before["wins"], before["ties"], before["losses"]],
                "new_window_level_wins_ties_losses": [after["wins"], after["ties"], after["losses"]],
                "new_paired_row_wins_ties_losses": [
                    after["paired_row_wins"], after["paired_row_ties"], after["paired_row_losses"]
                ],
                "old_sign_test_pvalue": before["sign_test_pvalue"],
                "new_window_sign_test_pvalue": after["sign_test_pvalue"],
                "old_holm_sign_test_pvalue": before["holm_sign_test_pvalue"],
                "new_holm_sign_test_pvalue": after["holm_sign_test_pvalue"],
                "old_claim_status_with_correct_signed_reading": old_claims[identity]["status"],
                "new_claim_status": new_claims[identity]["status"],
                "ci_unchanged": [before["ci95_low"], before["ci95_high"]]
                == [after["ci95_low"], after["ci95_high"]],
            }
        )
    counts = Counter(row["status"] for row in new_claims.values())
    return (
        {
            "claim_map_version": "g14r21_signed_ci_once_v1",
            "classification_basis": "signed_positive_favors_candidate=true; signed CI is never direction-flipped again",
            "execution_completeness_gate_is_scientifically_separate": True,
            "counts": dict(sorted(counts.items())),
            "rows": [new_claims[key] for key in sorted(new_claims)],
        },
        {
            "difference_report_version": "g14r21_old_to_new_v1",
            "root_causes": [
                "old sign test counted seed/workflow/capacity paired rows instead of outer-window means",
                "old formal-gate claim helper applied metric direction after statistics had already signed every CI",
            ],
            "unchanged_rules": {
                "bootstrap_replicates": new["bootstrap_samples"],
                "bootstrap_seed": 1401,
                "ci": ["percentile_95", "BCa_95"],
                "effect_sizes": ["paired Cohen dz", "outer-window standardized mean delta"],
                "holm_family_size": len(new_rows),
            },
            "rows": differences,
        },
    )


def write_paper_table(output_root: Path, payload: Mapping[str, Any]) -> None:
    fields = [
        "baseline_agent", "metric", "higher_is_better",
        "raw_mean_delta_candidate_minus_baseline", "mean_delta", "ci95_low", "ci95_high",
        "ci95_method", "available_paired_count", "total_pair_count", "outer_cluster_count",
        "wins", "ties", "losses", "sign_test_denominator", "sign_test_pvalue",
        "holm_sign_test_pvalue", "holm_preregistered_family_size",
    ]
    rows = payload["rows"]
    csv_path = output_root / "paper_primary_comparisons.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    lines = [
        "# Corrected primary paired comparisons",
        "",
        "`raw_mean_delta_candidate_minus_baseline` is always candidate minus baseline. "
        "`mean_delta` and its CI are signed exactly once so positive favors the candidate. "
        "The exact sign test uses outer-window means; ties are excluded from its denominator.",
        "",
        "| Baseline | Endpoint | Raw Δ | Signed Δ [BCa 95% CI] | W/T/L windows | Exact p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline_agent']} | {row['metric']} | {row['raw_mean_delta_candidate_minus_baseline']:.6f} "
            f"| {row['mean_delta']:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] "
            f"| {row['wins']}/{row['ties']}/{row['losses']} | {row['sign_test_pvalue']:.6f} "
            f"| {row['holm_sign_test_pvalue']:.6f} |"
        )
    (output_root / "paper_primary_comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_plan(split: str) -> tuple[Path, list[dict[str, Any]]]:
    path = ROOT / f"configs/experiment/typed_model_cache_formal_protocol_v1_20260820/{split}_window_plan.json"
    return path, load_json(path)["selected_window_plan"]


def overlaps(left: Mapping[str, Any], right: Mapping[str, Any], prefix: str) -> bool:
    if left["source_segment_run_id"] != right["source_segment_run_id"]:
        return False
    return int(left[f"{prefix}_start"]) <= int(right[f"{prefix}_end"]) and int(right[f"{prefix}_start"]) <= int(left[f"{prefix}_end"])


def audit_intervals() -> dict[str, Any]:
    plans = {split: load_plan(split) for split in SPLITS}
    conflicts: list[dict[str, Any]] = []
    minimum_gaps: dict[str, int | None] = {"raw_frame": None, "raw_time": None}
    all_rows: list[tuple[str, Mapping[str, Any]]] = [
        (split, row) for split, (_, rows) in plans.items() for row in rows
    ]
    for index, (left_split, left) in enumerate(all_rows):
        for right_split, right in all_rows[index + 1 :]:
            if left["source_segment_run_id"] != right["source_segment_run_id"]:
                continue
            for prefix in ("raw_frame", "raw_time"):
                if overlaps(left, right, prefix):
                    conflicts.append(
                        {
                            "kind": prefix,
                            "left_split": left_split,
                            "left_window_id": left["window_id"],
                            "right_split": right_split,
                            "right_window_id": right["window_id"],
                        }
                    )
                else:
                    earlier, later = sorted((left, right), key=lambda row: int(row[f"{prefix}_start"]))
                    gap = int(later[f"{prefix}_start"]) - int(earlier[f"{prefix}_end"]) - 1
                    current = minimum_gaps[prefix]
                    minimum_gaps[prefix] = gap if current is None else min(current, gap)
    return {
        "interval_audit_version": "g14r21_preopen_metadata_only_v1",
        "performance_fields_read": False,
        "policy_or_outcome_consumer_invoked": False,
        "split_counts": {split: len(rows) for split, (_, rows) in plans.items()},
        "plan_files": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, (path, _) in plans.items()
        },
        "checked_identity": "source_segment_run_id",
        "checked_intervals": ["raw_frame_start/raw_frame_end", "raw_time_start/raw_time_end"],
        "conflicts": conflicts,
        "minimum_nonoverlap_gaps": minimum_gaps,
        "passed": not conflicts,
    }


def build_holdout_application(formal_root: Path, interval_audit: Mapping[str, Any]) -> dict[str, Any]:
    seal_path = ROOT / "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/holdout_seal_record.json"
    seal = load_json(seal_path)
    protocol_path = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_v2_9_manifest.json"
    protocol = load_json(protocol_path)
    source_ref_path = formal_root / "evaluation_model_source_reference.json"
    source_ref = load_json(source_ref_path)
    capabilities = get_protocol_capabilities(protocol["typed_model_cache_formal_protocol_version"])
    checkpoint_manifests = {}
    source_run = Path(source_ref["source_run_root"])
    for capacity in ("constrained_288mb", "medium_576mb", "relaxed_864mb"):
        for kind in ("seed_checkpoint_manifest", "checkpoint_provenance_manifest"):
            path = source_run / "checkpoint_manifests" / capacity / f"{kind}.json"
            checkpoint_manifests[f"{capacity}.{kind}"] = {
                "path": str(path), "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
    blockers = [
        "one-time execution token is not issued",
        "Protocol 2.9 capability registry explicitly exposes holdout_capability=false",
        "no dedicated end-to-end public holdout runner/command plan is frozen",
        "exact output root, append-only opening ledger, completion receipt, and integrity consumer are not frozen as one transaction",
        "exact run quantity and tested wall-clock upper bound are not frozen",
    ]
    return {
        "holdout_application_version": "g14r21_unsigned_v1",
        "status": "NOT_ISSUED_BLOCKED",
        "grant_signed": False,
        "execution_authorized": False,
        "holdout_opened": False,
        "holdout_consumed_permanently": False,
        "review_scope": "seal/split metadata and code only; no performance labels",
        "seal": {
            "path": str(seal_path), "sha256": sha256_file(seal_path),
            "semantic_sha256": seal["hashes"]["semantic_sha256"],
            "sealed": seal["sealed"], "opened": seal["opened"],
            "consumed_permanently": seal["consumed_permanently"],
            "token_status": seal["one_time_execution_token_status"],
        },
        "interval_isolation": dict(interval_audit),
        "frozen_bindings": {
            "candidate_source_reference": {"path": str(source_ref_path), "sha256": sha256_file(source_ref_path)},
            "scientific_commit": source_ref["scientific_commit"],
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path), "semantic_sha256": protocol["hashes"]["semantic_sha256"]},
            "checkpoint_freeze_sha256": source_ref["checkpoint_freeze_sha256"],
            "generated_checkpoint_registry_file_sha256": source_ref["source_generated_registry_file_sha256"],
            "models_declared": source_ref["model_count"],
            "checkpoint_manifests": checkpoint_manifests,
            "capacity_labels": [row["stratum"] for row in protocol["typed_catalog_and_capacity"]["capacity_strata"]],
            "catalog_fingerprint": protocol["typed_catalog_and_capacity"]["catalog_fingerprint"],
            "training_budget": protocol["training_budget"],
            "statistics_and_claim_rule": "corrected G14R21 package must be committed and hash-bound before any opening",
        },
        "engineering_findings": {
            "current_protocol_holdout_capability": capabilities.holdout_capability,
            "ordinary_formal_runner_rejects_holdout": True,
            "metadata_level_append_only_record_primitive_exists": True,
            "cross_checkout_absolute_paths_present": True,
            "dedicated_public_holdout_orchestrator_exists": False,
        },
        "authorization_scope_required": {
            "one_time_only": True,
            "fixed_candidate_checkpoint_capacity_budget_family": True,
            "no_training_selection_tuning_or_window_choice": True,
            "stop_on_nonzero_without_automatic_retry": True,
            "infrastructure_retry_only_under_existing_seal_conditions": True,
        },
        "runtime_estimate": {
            "historical_formal_cache_policy_plus_controller_lower_bound": "5h12m",
            "statistics_and_gate_additional_lower_bound": "15m",
            "tested_upper_bound": None,
            "calendar_promise": None,
        },
        "blockers": blockers,
        "application_eligible": False,
        "ready": False,
    }


def find_typed_semantics(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        if payload.get("family") == "ablation" and payload.get("parameter") == "typed_semantics":
            return payload
        for value in payload.values():
            found = find_typed_semantics(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_typed_semantics(value)
            if found is not None:
                return found
    return None


def build_scientific_limitations(formal_root: Path, corrected: Mapping[str, Any]) -> dict[str, Any]:
    handoff_reference = load_json(formal_root / "post_ablation_handoff_reference.json")
    handoff = load_json(Path(handoff_reference["path"]))
    controller_paths: dict[str, Path] = {}
    for cell in handoff["cells"]:
        if cell["phase"] == "formal_controller":
            matches = list(Path(cell["committed_path"]).rglob("benchmark_rows.csv"))
            if len(matches) != 1:
                raise ValueError("controller handoff cell must expose exactly one row file")
            controller_paths[cell["coordinates"]["capacity_label"]] = matches[0]
    rows_by_capacity = {
        capacity: list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
        for capacity, path in controller_paths.items()
    }
    pair_fields = ("agent_name", "seed", "window_id", "workflow_id")
    medium = {tuple(row[field] for field in pair_fields): row for row in rows_by_capacity["medium_576mb"]}
    relaxed = {tuple(row[field] for field in pair_fields): row for row in rows_by_capacity["relaxed_864mb"]}
    equality_metrics = (*PRIMARY_METRICS, "total_reward")
    capacity_differences = {
        metric: sum(medium[key].get(metric) != relaxed[key].get(metric) for key in medium)
        for metric in equality_metrics
    }
    masks: dict[str, set[tuple[str, str, str, str]]] = {}
    for capacity, rows in rows_by_capacity.items():
        for row in rows:
            masks.setdefault(row["agent_name"], set())
            if row["end_to_end_workflow_delay"] != "":
                masks[row["agent_name"]].add((capacity, row["seed"], row["window_id"], row["workflow_id"]))
    unique_masks = {tuple(sorted(mask)) for mask in masks.values()}
    candidate_mask = masks["sa_ghmappo"]
    typed = find_typed_semantics(
        load_json(ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_v2_9_manifest.json")
    )
    if typed is None:
        raise ValueError("typed semantics preregistration is missing")
    executable = [row["value"] for row in typed["levels"] if row["status"] == "available"]
    unavailable = [row["value"] for row in typed["levels"] if row["status"] == "unavailable_pre_execution"]
    delay_rows = [row for row in corrected["rows"] if row["metric"] == "end_to_end_workflow_delay"]
    transfer_rows = [row for row in corrected["rows"] if row["metric"] == "transfer_mb_per_request"]
    return {
        "scientific_limitations_audit_version": "g14r21_v1",
        "delay": {
            "estimand": "conditional_on_completed_workflows",
            "candidate_available_pairs": len(candidate_mask),
            "candidate_total_pairs": sum(1 for rows in rows_by_capacity.values() for row in rows if row["agent_name"] == "sa_ghmappo"),
            "candidate_available_outer_windows": len({item[2] for item in candidate_mask}),
            "all_15_agents_share_identical_availability_mask": len(unique_masks) == 1,
            "all_14_pairwise_delay_effects_exactly_zero": all(
                row["mean_delta"] == 0 and row["ci95_low"] == 0 and row["ci95_high"] == 0
                for row in delay_rows
            ),
            "null_imputed_as_zero": False,
        },
        "capacity": {
            "medium_relaxed_pair_identity_equal": set(medium) == set(relaxed),
            "paired_row_count": len(medium),
            "differences_by_primary_reward_metric": capacity_differences,
            "non_discriminating_576mb_vs_864mb": all(value == 0 for value in capacity_differences.values()),
        },
        "tradeoff": {
            "transfer_candidate_adverse_ci_count": sum(row["ci95_high"] < 0 for row in transfer_rows),
            "transfer_mixed_ci_count": sum(row["ci95_low"] <= 0 <= row["ci95_high"] for row in transfer_rows),
            "service_and_continuity_claims_are_mixed": all(
                row["ci95_low"] <= 0 <= row["ci95_high"]
                for row in corrected["rows"]
                if row["metric"] != "transfer_mb_per_request"
            ),
            "interpretation": "sparse local service/readiness gain with transfer/backhaul cost; no Pareto dominance",
        },
        "typed_semantics_ablation": {
            "preregistered_levels": [row["value"] for row in typed["levels"]],
            "executable_levels": executable,
            "unavailable_pre_execution_levels": unavailable,
            "unavailable_count": len(unavailable),
        },
        "non_significance_is_equivalence": False,
    }


def source_inventory(formal_root: Path, corrected: Path, review: Path) -> dict[str, Any]:
    source_paths = [
        formal_root / "statistics/paired_statistics.json",
        formal_root / "formal_gate.json",
        formal_root / "artifact_integrity_manifest.json",
        formal_root / "phase_state.jsonl",
        formal_root / "cell_state.jsonl",
        formal_root / "evaluation_model_source_reference.json",
        formal_root / "resolved_execution_context.json",
        corrected,
        review,
        ROOT / "scripts/analyze_top_journal_statistics.py",
        ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py",
    ]
    old_stats = load_json(formal_root / "statistics/paired_statistics.json")
    source_paths.extend(Path(path) for path in old_stats["source_rows_path"])
    return {
        "sources": [
            {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None,
             "sha256": sha256_file(path) if path.is_file() else None}
            for path in source_paths
        ],
        "repository": {
            "checkout": str(ROOT), "git_commit": git("rev-parse", "HEAD"),
            "git_tree": git("rev-parse", "HEAD^{tree}"), "branch": git("branch", "--show-current"),
        },
        "environment": {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        },
    }


def write_integrity(output_root: Path) -> None:
    target = output_root / "artifact_integrity_manifest.json"
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != target:
            files.append({"path": path.relative_to(output_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(target, {"artifact_integrity_version": "g14r21_v1", "file_count": len(files), "files": files})


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    corrected_path = args.corrected_statistics.resolve()
    formal_root = args.formal_root.resolve()
    corrected = load_json(corrected_path)
    old = load_json(formal_root / "statistics/paired_statistics.json")
    claim_map, difference = build_claim_and_diff(old, corrected)
    write_json(output_root / "corrected_claim_map.json", claim_map)
    write_json(output_root / "old_to_new_difference.json", difference)
    write_paper_table(output_root, corrected)
    interval_audit = audit_intervals()
    write_json(output_root / "holdout_interval_audit.json", interval_audit)
    application = build_holdout_application(formal_root, interval_audit)
    write_json(output_root / "holdout_application_unsigned.json", application)
    limitations = build_scientific_limitations(formal_root, corrected)
    write_json(output_root / "scientific_limitations.json", limitations)
    inventory = source_inventory(formal_root, corrected_path, args.g14a01_review.resolve())
    write_json(output_root / "source_provenance.json", inventory)
    summary = {
        "analysis_contract_version": "g14r21_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_at": "2026-09-23",
        "literature_cutoff": "2026-09-21",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": formal_root.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "evidence_level": "E3_REPRODUCED_FOR_CORRECTED_FORMAL_STATISTICS_ONLY; holdout absent",
        "verdict": "formal statistics correction accepted if integrity/tests pass; paper-ready remains Unverifiable",
        "claim_counts": claim_map["counts"],
        "holm_significant_count_alpha_0_05": sum(float(row["holm_sign_test_pvalue"]) < 0.05 for row in corrected["rows"]),
        "holm_family_size": len(corrected["rows"]),
        "execution_gate": {"path": str(formal_root / "formal_gate.json"), "meaning": "completeness only; no performance threshold"},
        "scientific_claim_classification": {"path": str(output_root / "corrected_claim_map.json"), "meaning": "signed CI classification; separate from execution gate"},
        "limitations_preserved": [
            "delay is conditional on completed workflows; null is never zero-imputed",
            "delay has 9/12 effective outer windows in the frozen formal result",
            "576 MB and 864 MB primary outcomes are non-discriminating in the frozen formal result",
            "local readiness/handoff gains trade against transfer/backhaul cost",
            "four of six typed-semantics ablations were unavailable before execution",
            "non-significance is not equivalence",
        ],
        "holdout_application": {"status": application["status"], "ready": False, "blocker_count": len(application["blockers"])},
        "prohibited_work": {"training": 0, "model_selection": 0, "formal_rollout": 0, "holdout_policy_runs": 0, "holdout_performance_labels_read": 0},
    }
    write_json(output_root / "analysis_summary.json", summary)
    write_json(
        output_root / "command_environment.json",
        {"command": [sys.executable, *sys.argv], "cwd": os.getcwd(), "started_and_completed_in_process": True, "exit_code": 0, **inventory["environment"]},
    )
    write_integrity(output_root)
    print(json.dumps({"status": "pass", "output_root": str(output_root), "claim_counts": claim_map["counts"], "holdout_ready": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
