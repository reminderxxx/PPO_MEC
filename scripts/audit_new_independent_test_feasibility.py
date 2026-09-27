"""Metadata-only feasibility audit for a prospective new temporal test split.

The audit never imports performance rows and never runs an agent.  It extends
the frozen G14B history ledger with later selected-window metadata, applies a
time embargo, checks vehicle recurrence from raw mobility identity columns,
and emits a non-executable proposal only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_protocol import (
    canonical_sha256,
    classify_history_purpose,
    discover_historical_plan_files,
    extract_selected_window_plan,
    interval_relation,
    normalized_segment,
    result_blind_window_projection,
)


BASELINE_ARTIFACT = (
    ROOT
    / "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1"
)
DEFAULT_SOURCE = (
    ROOT
    / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
)
AUDIT_VERSION = "g14r23_new_independent_test_feasibility_v1.0.0"
SELECTION_SEED = 1423
TEMPORAL_EMBARGO_FRAMES = 24
PROPOSED_WINDOW_COUNT = 12
MINIMUM_DISTINCT_SEGMENT_RUNS = 3
MAXIMUM_WINDOWS_PER_SEGMENT_RUN = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", default=str(ROOT))
    parser.add_argument("--source-path", default=str(DEFAULT_SOURCE))
    parser.add_argument("--baseline-artifact-root", default=str(BASELINE_ARTIFACT))
    parser.add_argument(
        "--output-root",
        default=str(
            ROOT
            / "artifacts/analysis/g14r23_holdout_interface_and_independent_test_20260927"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def explicit_interval(window: Mapping[str, Any]) -> dict[str, Any] | None:
    item = result_blind_window_projection(window)
    required = (
        "source_segment_id",
        "source_segment_run_id",
        "raw_frame_start",
        "raw_frame_end",
        "raw_time_start",
        "raw_time_end",
    )
    if any(item.get(field) is None for field in required):
        return None
    return {
        "window_id": str(item.get("window_id") or "unknown"),
        "source_segment_id": normalized_segment(item["source_segment_id"]),
        "source_segment_run_id": str(item["source_segment_run_id"]),
        "raw_frame_start": int(item["raw_frame_start"]),
        "raw_frame_end": int(item["raw_frame_end"]),
        "raw_time_start": int(item["raw_time_start"]),
        "raw_time_end": int(item["raw_time_end"]),
        "sampling_interval": int(item.get("sampling_interval") or 100),
        "unknown_interval_flag": False,
        "conservative_exclusion_scope": [],
    }


def interval_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        item["source_segment_run_id"],
        int(item["raw_frame_start"]),
        int(item["raw_frame_end"]),
        int(item["raw_time_start"]),
        int(item["raw_time_end"]),
    )


def discover_later_usage(
    history_root: Path,
    baseline_registry: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_paths = {
        str(path)
        for record in baseline_registry["records"]
        for path in record.get("evidence_paths", [])
    }
    baseline_window_ids = {
        str(window_id)
        for record in baseline_registry["records"]
        for window_id in record.get("window_ids", [])
    }
    candidate_by_id = {str(item["window_id"]): dict(item) for item in candidates}
    paths = discover_historical_plan_files(history_root)
    rows: list[dict[str, Any]] = []
    explicit_by_window_id: dict[str, dict[str, Any]] = {}
    raw_references: list[dict[str, Any]] = []
    for path in paths:
        relative = relative_or_absolute(path, history_root)
        windows = extract_selected_window_plan(path)
        for window in windows:
            projected = result_blind_window_projection(window)
            record = explicit_interval(projected)
            window_id = str(projected.get("window_id") or "unknown")
            if record is not None:
                explicit_by_window_id.setdefault(window_id, record)
            raw_references.append(
                {
                    "path": relative,
                    "baseline_covered": relative in baseline_paths,
                    "window_id": window_id,
                    "purposes": classify_history_purpose(relative),
                    "explicit": record,
                    "projected": projected,
                }
            )
    unresolved_new = []
    duplicate_baseline_reference_count = 0
    synthetic_reference_count = 0
    known_non_i80_unresolved_reference_count = 0
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for reference in raw_references:
        if reference["baseline_covered"]:
            continue
        record = reference["explicit"]
        resolution = "explicit_raw_identity"
        if record is None:
            record = explicit_by_window_id.get(reference["window_id"])
            resolution = "same_window_id_explicit_identity"
        if record is None:
            candidate = candidate_by_id.get(reference["window_id"])
            record = explicit_interval(candidate or {})
            resolution = "frozen_candidate_identity"
        if record is None:
            projected = reference["projected"]
            segment = normalized_segment(projected.get("source_segment_id"))
            segment_run = str(projected.get("source_segment_run_id") or "")
            window_class = str(projected.get("window_class") or "")
            if segment_run == "synthetic" or window_class == "test_only":
                synthetic_reference_count += 1
                continue
            if reference["window_id"] in baseline_window_ids:
                duplicate_baseline_reference_count += 1
                continue
            if segment not in {"unknown", "i_80"}:
                known_non_i80_unresolved_reference_count += 1
                continue
            unresolved_new.append(
                {
                    "path": reference["path"],
                    "window_id": reference["window_id"],
                    "projected_metadata": reference["projected"],
                    "purposes": reference["purposes"],
                }
            )
            continue
        key = interval_key(record)
        target = by_key.setdefault(
            key,
            {
                **record,
                "evidence_paths": [],
                "window_ids": [],
                "purposes": [],
                "resolution_methods": [],
            },
        )
        for field, value in (
            ("evidence_paths", reference["path"]),
            ("window_ids", reference["window_id"]),
            ("resolution_methods", resolution),
        ):
            if value not in target[field]:
                target[field].append(value)
        for purpose in reference["purposes"]:
            if purpose not in target["purposes"]:
                target["purposes"].append(purpose)
    for row in by_key.values():
        for field in ("evidence_paths", "window_ids", "purposes", "resolution_methods"):
            row[field].sort()
        rows.append(row)
    rows.sort(key=lambda item: interval_key(item))
    return {
        "discovered_plan_or_result_file_count": len(paths),
        "raw_selected_window_reference_count": len(raw_references),
        "baseline_evidence_path_count": len(baseline_paths),
        "later_or_uncovered_unique_interval_count": len(rows),
        "later_or_uncovered_usage_records": rows,
        "unresolved_later_reference_count": len(unresolved_new),
        "unresolved_later_references": unresolved_new,
        "duplicate_baseline_reference_count": duplicate_baseline_reference_count,
        "synthetic_reference_count": synthetic_reference_count,
        "known_non_i80_unresolved_reference_count": (
            known_non_i80_unresolved_reference_count
        ),
        "performance_fields_decoded": False,
        "parser_scope": "selected_window_plan/selected_windows identity metadata only",
    }


def candidate_temporal_filter(
    candidates: list[dict[str, Any]],
    used: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible = []
    reasons: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        conflicts = []
        for record in used:
            relation = interval_relation(
                candidate,
                record,
                minimum_gap_frames=TEMPORAL_EMBARGO_FRAMES,
            )
            if relation["classification"] != "safe":
                conflicts.append(relation["classification"])
        if conflicts:
            for reason in set(conflicts):
                reasons[reason] += 1
            continue
        eligible.append(dict(candidate))
    return eligible, dict(sorted(reasons.items()))


def vehicle_sets_for_intervals(
    source_path: Path,
    candidates: list[dict[str, Any]],
    used: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[tuple[Any, ...], set[str]], dict[str, Any]]:
    candidate_sets = {str(item["window_id"]): set() for item in candidates}
    used_sets = {interval_key(item): set() for item in used}
    lookup: defaultdict[tuple[str, int], list[tuple[str, Any]]] = defaultdict(list)
    for item in candidates:
        for raw_time in range(
            int(item["raw_time_start"]),
            int(item["raw_time_end"]) + 1,
            int(item.get("sampling_interval") or 100),
        ):
            lookup[(str(item["source_segment_id"]), raw_time)].append(
                ("candidate", str(item["window_id"]))
            )
    for item in used:
        for raw_time in range(
            int(item["raw_time_start"]),
            int(item["raw_time_end"]) + 1,
            int(item.get("sampling_interval") or 100),
        ):
            lookup[(str(item["source_segment_id"]), raw_time)].append(
                ("used", interval_key(item))
            )
    matched_rows = 0
    scanned_rows = 0
    for chunk in pd.read_csv(
        source_path,
        usecols=["Vehicle_ID", "Location", "Global_Time"],
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    ):
        scanned_rows += len(chunk)
        segments = chunk["Location"].map(normalized_segment)
        times = pd.to_numeric(
            chunk["Global_Time"].str.replace(",", "", regex=False), errors="raise"
        ).astype("int64")
        mask = pd.Series(
            [
                (str(segment), int(raw_time)) in lookup
                for segment, raw_time in zip(segments, times)
            ],
            index=chunk.index,
        )
        if not mask.any():
            continue
        for index in chunk.index[mask]:
            key = (str(segments.loc[index]), int(times.loc[index]))
            vehicle_id = str(chunk.loc[index, "Vehicle_ID"])
            matched_rows += 1
            for kind, identity in lookup[key]:
                if kind == "candidate":
                    candidate_sets[identity].add(vehicle_id)
                else:
                    used_sets[identity].add(vehicle_id)
    return candidate_sets, used_sets, {
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "scanned_row_count": scanned_rows,
        "matched_identity_row_count": matched_rows,
        "columns_read": ["Vehicle_ID", "Location", "Global_Time"],
        "performance_fields_read": False,
    }


def deterministic_proposal(
    candidates: list[dict[str, Any]],
    candidate_vehicle_sets: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    by_run: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        ranked = dict(item)
        ranked["proposal_rank_sha256"] = hashlib.sha256(
            (
                f"{SELECTION_SEED}|{item['source_segment_run_id']}|"
                f"{item['raw_frame_start']}|{item['raw_time_start']}"
            ).encode("utf-8")
        ).hexdigest()
        by_run[str(item["source_segment_run_id"])].append(ranked)
    for values in by_run.values():
        values.sort(key=lambda item: (item["proposal_rank_sha256"], item["window_id"]))
    selected = []
    selected_vehicle_ids: set[str] = set()
    selected_by_run: defaultdict[str, int] = defaultdict(int)
    run_ids = sorted(by_run)
    cursor = 0
    while len(selected) < PROPOSED_WINDOW_COUNT and any(by_run.values()):
        run_id = run_ids[cursor % len(run_ids)]
        if by_run[run_id]:
            candidate = by_run[run_id].pop(0)
            vehicle_ids = candidate_vehicle_sets[str(candidate["window_id"])]
            if (
                selected_by_run[run_id] < MAXIMUM_WINDOWS_PER_SEGMENT_RUN
                and not vehicle_ids.intersection(selected_vehicle_ids)
            ):
                selected.append(candidate)
                selected_vehicle_ids.update(vehicle_ids)
                selected_by_run[run_id] += 1
        cursor += 1
    return selected


def main() -> None:
    args = parse_args()
    history_root = Path(args.history_root).resolve()
    source_path = Path(args.source_path).resolve()
    baseline_root = Path(args.baseline_artifact_root).resolve()
    output_root = Path(args.output_root).resolve()
    baseline_registry = read_json(baseline_root / "historical_window_usage_registry.json")
    candidate_inventory = read_json(baseline_root / "candidate_window_inventory.json")
    candidates = [dict(item) for item in candidate_inventory["candidates"]]
    history = discover_later_usage(history_root, baseline_registry, candidates)
    later_used = history["later_or_uncovered_usage_records"]
    temporal_candidates, temporal_exclusions = candidate_temporal_filter(
        candidates, later_used
    )
    candidate_sets, used_sets, vehicle_scan = vehicle_sets_for_intervals(
        source_path, temporal_candidates, later_used
    )
    all_used_vehicle_ids = set().union(*used_sets.values()) if used_sets else set()
    vehicle_eligible = []
    vehicle_rows = []
    for item in temporal_candidates:
        ids = candidate_sets[str(item["window_id"])]
        overlap = ids.intersection(all_used_vehicle_ids)
        vehicle_rows.append(
            {
                "window_id": item["window_id"],
                "vehicle_id_count": len(ids),
                "used_vehicle_overlap_count": len(overlap),
            }
        )
        if ids and not overlap:
            vehicle_eligible.append(item)
    unresolved_blocks_i80 = any(
        normalized_segment(row["projected_metadata"].get("source_segment_id"))
        in {"unknown", "i_80"}
        for row in history["unresolved_later_references"]
    )
    candidate_preview = (
        deterministic_proposal(vehicle_eligible, candidate_sets)
        if not unresolved_blocks_i80 and len(vehicle_eligible) >= PROPOSED_WINDOW_COUNT
        else []
    )
    proposed_vehicle_pairs = []
    proposed_temporal_pairs = []
    for index, left in enumerate(candidate_preview):
        for right in candidate_preview[index + 1 :]:
            overlap_count = len(
                candidate_sets[str(left["window_id"])].intersection(
                    candidate_sets[str(right["window_id"])]
                )
            )
            proposed_vehicle_pairs.append(
                {
                    "left_window_id": left["window_id"],
                    "right_window_id": right["window_id"],
                    "vehicle_overlap_count": overlap_count,
                }
            )
            relation = interval_relation(
                left,
                right,
                minimum_gap_frames=TEMPORAL_EMBARGO_FRAMES,
            )
            proposed_temporal_pairs.append(
                {
                    "left_window_id": left["window_id"],
                    "right_window_id": right["window_id"],
                    **relation,
                }
            )
    if any(row["classification"] != "safe" for row in proposed_temporal_pairs):
        candidate_preview = []
        proposed_vehicle_pairs = []
        proposed_temporal_pairs = []
    preview_run_counts: defaultdict[str, int] = defaultdict(int)
    for item in candidate_preview:
        preview_run_counts[str(item["source_segment_run_id"])] += 1
    reliable_correlation_coverage = (
        len(candidate_preview) == PROPOSED_WINDOW_COUNT
        and len(preview_run_counts) >= MINIMUM_DISTINCT_SEGMENT_RUNS
        and max(preview_run_counts.values(), default=0)
        <= MAXIMUM_WINDOWS_PER_SEGMENT_RUN
    )
    proposed = candidate_preview if reliable_correlation_coverage else []
    feasible = len(proposed) == PROPOSED_WINDOW_COUNT
    report = {
        "audit_version": AUDIT_VERSION,
        "reviewed_at": "2026-09-27",
        "literature_cutoff": "2026-09-27",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": output_root.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "evidence_level": "E2_METADATA_AUDITED_NO_NEW_PERFORMANCE_EXECUTION",
        "scope": {
            "metadata_only": True,
            "agent_or_policy_executed": False,
            "performance_fields_read": False,
            "sealed_holdout_outcomes_read": False,
            "new_test_run_executed": False,
        },
        "baseline_history": {
            "registry_semantic_sha256": baseline_registry["hashes"]["semantic_sha256"],
            "summary": baseline_registry["summary"],
            "candidate_inventory_semantic_sha256": candidate_inventory["hashes"][
                "semantic_sha256"
            ],
            "candidate_count": len(candidates),
        },
        "full_history_extension": history,
        "correlation_audit": {
            "temporal_embargo_frames": TEMPORAL_EMBARGO_FRAMES,
            "temporal_eligible_count": len(temporal_candidates),
            "temporal_exclusion_counts": temporal_exclusions,
            "vehicle_zero_recurrence_eligible_count": len(vehicle_eligible),
            "minimum_distinct_segment_runs_required": (
                MINIMUM_DISTINCT_SEGMENT_RUNS
            ),
            "maximum_windows_per_segment_run": MAXIMUM_WINDOWS_PER_SEGMENT_RUN,
            "candidate_preview_count_before_run_correlation_gate": len(
                candidate_preview
            ),
            "candidate_preview_segment_run_counts": dict(
                sorted(preview_run_counts.items())
            ),
            "reliable_correlation_coverage_passed": reliable_correlation_coverage,
            "pairwise_vehicle_disjoint_proposable_count": len(proposed),
            "candidate_preview_pairwise_vehicle_overlap_rows": proposed_vehicle_pairs,
            "candidate_preview_pairwise_temporal_relation_rows": proposed_temporal_pairs,
            "vehicle_rows": vehicle_rows,
            "vehicle_scan": vehicle_scan,
            "unresolved_later_reference_blocks_i80": unresolved_blocks_i80,
            "residual_dependence": [
                "all candidates remain in the same NGSIM I-80 source and road segment",
                "candidate windows may share a capture session/segment-run even when time and vehicles differ",
                "the Alibaba workflow source and controlled typed catalog remain unchanged",
                "therefore temporal non-overlap and zero observed vehicle recurrence do not establish external-dataset independence",
            ],
        },
        "verdict": (
            "FEASIBLE_AS_PROSPECTIVE_INTERNAL_TEMPORAL_TEST_NOT_PRISTINE_EXTERNAL_HOLDOUT"
            if feasible
            else "NO_RELIABLE_UNUSED_RANGE_ESTABLISHED"
        ),
        "proposal": {
            "status": "pending_independent_approval" if feasible else "not_created",
            "execution_authorized": False,
            "grant_signed": False,
            "seal_created": False,
            "selection_seed": SELECTION_SEED,
            "selection_rule": (
                "Start from the 2026-08-20 result-blind 579-window inventory; exclude every later/full-history used or allocated interval with a 24-frame embargo; require non-empty vehicle coverage, zero Vehicle_ID recurrence against every used I-80 interval, zero Vehicle_ID recurrence among selected proposal windows, at least three distinct segment-runs, and at most four windows per run; rank by SHA-256(seed, segment-run, raw-frame-start, raw-time-start); take round-robin across segment-runs until 12. No outcome, reward, cache, checkpoint ranking, or agent result is an input."
            ),
            "proposed_windows": proposed,
            "required_binding_plan": {
                "new_split_manifest": "create-only after approval; bind exact proposed window identities and full-history audit hash",
                "new_seal": "create-only after approval; unopened and one-time consumed semantics",
                "frozen_model": "bind the existing immutable candidate checkpoint manifest and every checkpoint SHA-256; no training, reselection, or replacement",
                "corrected_statistics": "bind the corrected signed claim rule, window-level sign test, 84-family Holm procedure, capacity-separated reporting, and conditional-delay availability rules before opening",
                "execution": "bind dedicated authorization, Git commit, exact benchmark argv, output root/run ID, window contract/plan hashes, and append-only opening record",
            },
            "forbidden_before_approval": [
                "sign grant or issue one-time token",
                "create an executable seal",
                "run any agent/checkpoint on proposed windows",
                "inspect outcomes and then change the window set",
                "change training, hyperparameters, checkpoint, baseline family, or original results",
            ],
        },
        "scientific_limit": (
            "Even if approved, this is an internal temporal confirmation split, not a pristine external holdout and not paper-ready evidence by itself."
        ),
    }
    report["report_semantic_sha256"] = canonical_sha256(report)
    write_json(output_root / "new_independent_test_feasibility.json", report)
    summary = {
        "status": "completed",
        "report_path": str(output_root / "new_independent_test_feasibility.json"),
        "verdict": report["verdict"],
        "proposed_window_count": len(proposed),
        "execution_authorized": False,
        "new_test_run_executed": False,
        "report_semantic_sha256": report["report_semantic_sha256"],
    }
    write_json(output_root / "completion_receipt.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
