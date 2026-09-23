#!/usr/bin/env python3
"""Build a capacity-aware reanalysis from frozen immutable formal rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manage_typed_model_cache_formal_artifacts import _claim_evidence_rows
from src.runtime.generated_checkpoint_resources import CAPACITY_MB
from src.runtime.restricted_recovery import validate_recovery_handoff_manifest


FORMAL_ROOT_DEFAULT = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_evaluation_only/"
    "typed_model_cache_post_ablation_20260921_g14e07_pending"
)
OLD_STATISTICS_DEFAULT = ROOT / (
    "artifacts/analysis/typed_model_cache_g14r21_corrected_statistics_holdout_audit_"
    "20260923_v1/corrected_statistics/paired_statistics.json"
)
METRICS = (
    "full_service_ready_byte_hit_rate",
    "joint_base_adapter_hit_rate",
    "full_service_ready_request_rate",
    "transfer_mb_per_request",
    "workflow_continuity_rate",
    "end_to_end_workflow_delay",
)
PAIR_KEYS = ("seed", "window_id", "workflow_id", "capacity_label")
OUTER_KEYS = ("source_segment_run_id", "window_id")
INNER_KEYS = ("seed", "workflow_id", "capacity_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT_DEFAULT)
    parser.add_argument("--old-statistics", type=Path, default=OLD_STATISTICS_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def manifest_binding(root: Path, relative: str) -> dict[str, Any]:
    manifest_path = root / "artifact_integrity_manifest.json"
    manifest = load_json(manifest_path)
    matches = [row for row in manifest.get("files", []) if row.get("path") == relative]
    if len(matches) != 1:
        raise ValueError(f"producer manifest does not bind exactly one {relative}: {root}")
    target = root / relative
    row = matches[0]
    if (
        not target.is_file()
        or target.is_symlink()
        or target.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(target) != row.get("sha256")
    ):
        raise ValueError(f"producer manifest byte validation failed: {target}")
    return {
        "path": str(target),
        "sha256": row["sha256"],
        "size_bytes": row["size_bytes"],
        "producer_manifest_path": str(manifest_path),
        "producer_manifest_sha256": sha256_file(manifest_path),
    }


def controller_sources(formal_root: Path, old: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference_path = formal_root / "post_ablation_handoff_reference.json"
    reference = load_json(reference_path)
    handoff_path = Path(str(reference.get("path", ""))).resolve()
    if not handoff_path.is_file() or handoff_path.is_symlink():
        raise ValueError("post-ablation handoff reference is missing or symlinked")
    handoff = load_json(handoff_path)
    handoff_audit = validate_recovery_handoff_manifest(handoff)
    if handoff_audit["handoff_sha256"] != reference.get("sha256"):
        raise ValueError("post-ablation handoff canonical identity mismatch")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in handoff.get("cells", []):
        if cell.get("phase") != "formal_controller":
            continue
        coordinates = cell.get("coordinates")
        if not isinstance(coordinates, dict):
            raise ValueError("formal controller cell lacks coordinates")
        capacity = coordinates.get("capacity_label")
        resource_id = coordinates.get("runtime_config_resource_id")
        if capacity not in CAPACITY_MB or resource_id != f"runtime_config.{capacity}":
            raise ValueError("formal controller capacity/resource identity is invalid")
        if capacity in seen:
            raise ValueError(f"duplicate formal controller capacity: {capacity}")
        seen.add(str(capacity))
        committed = Path(str(cell.get("committed_path", ""))).resolve()
        rows_binding = manifest_binding(committed, "benchmark_rows.csv")
        aggregate_binding = manifest_binding(committed, "aggregate_summary.json")
        run_manifest_binding = manifest_binding(committed, "run_manifest.json")
        aggregate = load_json(Path(aggregate_binding["path"]))
        runtime_capacity = aggregate.get("resolved_model_cache_runtime", {}).get(
            "cache_capacity_profile", {}
        )
        if (
            runtime_capacity.get("enabled") is not True
            or runtime_capacity.get("unit") != "mb"
            or float(runtime_capacity.get("capacity_mb")) != float(CAPACITY_MB[str(capacity)])
        ):
            raise ValueError(f"runtime capacity mismatch for {capacity}")
        window_map: dict[str, str] = {}
        for row in aggregate.get("selected_window_plan", []):
            window = str(row.get("window_id", ""))
            segment = str(row.get("source_segment_run_id", ""))
            if not window or not segment or window in window_map:
                raise ValueError(f"invalid selected-window identity for {capacity}")
            window_map[window] = segment
        result.append(
            {
                "capacity_label": capacity,
                "runtime_config_resource_id": resource_id,
                "committed_path": str(committed),
                "rows": rows_binding,
                "aggregate": aggregate_binding,
                "run_manifest": run_manifest_binding,
                "window_to_source_segment_run_id": window_map,
                "capacity_mb": CAPACITY_MB[str(capacity)],
                "identity_source": (
                    "hash_verified_post_ablation_handoff.coordinates + "
                    "producer_manifest + runtime contract"
                ),
            }
        )
    if seen != set(CAPACITY_MB):
        raise ValueError(f"capacity coverage mismatch: {sorted(seen)}")
    old_sources = {str(Path(path).resolve()) for path in old.get("source_rows_path", [])}
    discovered = {row["rows"]["path"] for row in result}
    if old_sources != discovered:
        raise ValueError("G14R21 source rows differ from verified controller handoff rows")
    return sorted(result, key=lambda row: list(CAPACITY_MB).index(row["capacity_label"]))


def augment_inputs(output_root: Path, sources: list[dict[str, Any]]) -> list[Path]:
    output_paths: list[Path] = []
    for source in sources:
        source_path = Path(source["rows"]["path"])
        before = sha256_file(source_path)
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"source CSV has no header: {source_path}")
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
        for field in (
            "capacity_label",
            "source_segment_run_id",
            "runtime_config_resource_id",
            "capacity_identity_source",
        ):
            if field not in fieldnames:
                fieldnames.append(field)
        seen_coordinates: set[tuple[str, ...]] = set()
        window_map = source["window_to_source_segment_run_id"]
        for row in rows:
            window = row.get("window_id", "")
            if window not in window_map:
                raise ValueError(f"row window is absent from validated producer plan: {window}")
            existing_capacity = row.get("capacity_label", "")
            if existing_capacity and existing_capacity != source["capacity_label"]:
                raise ValueError("source CSV has conflicting capacity identity")
            existing_segment = row.get("source_segment_run_id", "")
            if existing_segment and existing_segment != window_map[window]:
                raise ValueError("source CSV has conflicting source-segment identity")
            row["capacity_label"] = source["capacity_label"]
            row["source_segment_run_id"] = window_map[window]
            row["runtime_config_resource_id"] = source["runtime_config_resource_id"]
            row["capacity_identity_source"] = source["identity_source"]
            coordinate = tuple(row.get(field, "") for field in (*PAIR_KEYS, "agent_name"))
            if "" in coordinate or coordinate in seen_coordinates:
                raise ValueError(f"missing or duplicate augmented row coordinate: {coordinate}")
            seen_coordinates.add(coordinate)
        destination = output_root / "analysis_inputs" / f"{source['capacity_label']}.benchmark_rows.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if sha256_file(source_path) != before:
            raise ValueError("immutable source CSV changed during augmentation")
        output_paths.append(destination)
    return output_paths


def claims(payload: Mapping[str, Any]) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    rows = _claim_evidence_rows(payload)
    by_identity = {
        (str(row["baseline_agent"]), str(row["metric"])): str(row["status"])
        for row in rows
    }
    return by_identity, dict(sorted(Counter(by_identity.values()).items()))


def compare(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    old_rows = {(row["baseline_agent"], row["metric"]): row for row in old["rows"]}
    new_rows = {(row["baseline_agent"], row["metric"]): row for row in new["rows"]}
    if set(old_rows) != set(new_rows) or len(new_rows) != 84:
        raise ValueError("old/new comparison does not preserve the 84-item family")
    old_claims, old_counts = claims(old)
    new_claims, new_counts = claims(new)
    fields = (
        "total_pair_count",
        "available_paired_count",
        "availability_coverage",
        "outer_cluster_count",
        "inner_cluster_count",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "cohen_dz",
        "outer_cluster_cohen_d",
        "wins",
        "ties",
        "losses",
        "sign_test_pvalue_exact",
        "holm_sign_test_pvalue",
    )
    rows = []
    changed_fields: Counter[str] = Counter()
    for identity in sorted(new_rows):
        before, after = old_rows[identity], new_rows[identity]
        changes = {field: before.get(field) != after.get(field) for field in fields}
        changed_fields.update(field for field, changed in changes.items() if changed)
        rows.append(
            {
                "baseline_agent": identity[0],
                "metric": identity[1],
                "old": {field: before.get(field) for field in fields},
                "new": {field: after.get(field) for field in fields},
                "changed": changes,
                "old_claim": old_claims[identity],
                "new_claim": new_claims[identity],
                "claim_changed": old_claims[identity] != new_claims[identity],
            }
        )
    return {
        "comparison_version": "g14r22b_capacity_identity_old_new_v1",
        "old_applicability": (
            "audit-only: paired coordinates were separated by source file, while the declared "
            "capacity inner-cluster identity was absent and silently collapsed"
        ),
        "new_applicability": "capacity-aware formal reanalysis from immutable rows",
        "unchanged_scientific_budget": {
            "metrics": list(METRICS),
            "bootstrap_samples": new["bootstrap_samples"],
            "bootstrap_seed": 1401,
            "ci_method": new["requested_ci_method"],
            "holm_family_size": len(new_rows),
            "models_windows_and_rollouts_changed": False,
        },
        "old_claim_counts": old_counts,
        "new_claim_counts": new_counts,
        "claim_transition_counts": dict(
            sorted(Counter(f"{row['old_claim']}->{row['new_claim']}" for row in rows).items())
        ),
        "changed_field_row_counts": dict(sorted(changed_fields.items())),
        "claim_changed_count": sum(row["claim_changed"] for row in rows),
        "rows": rows,
    }


def write_integrity(output_root: Path) -> None:
    target = output_root / "artifact_integrity_manifest.json"
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != target:
            files.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        target,
        {
            "integrity_manifest_version": "g14r22b_capacity_statistics_v1",
            "file_count": len(files),
            "files": files,
        },
    )


def main() -> int:
    args = parse_args()
    if git("status", "--porcelain"):
        raise ValueError("capacity statistics package must be generated from a clean checkout")
    generator_repository = {
        "checkout": str(ROOT),
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "clean": True,
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    old_path = args.old_statistics.resolve()
    old = load_json(old_path)
    sources = controller_sources(args.formal_root.resolve(), old)
    augmented = augment_inputs(output_root, sources)
    baselines = old.get("pairwise_comparison_identity", {}).get("baseline_agent_order")
    if not isinstance(baselines, list) or len(baselines) != 14:
        raise ValueError("frozen baseline family is invalid")
    statistics_root = output_root / "capacity_aware_statistics"
    command = [
        sys.executable,
        str(ROOT / "scripts/analyze_top_journal_statistics.py"),
    ]
    for path in augmented:
        command.extend(("--rows_path", str(path)))
    command.extend(
        (
            "--candidate_agent", "sa_ghmappo",
            "--baseline_agents", *[str(value) for value in baselines],
            "--metrics", *METRICS,
            "--pair_keys", *PAIR_KEYS,
            "--outer_cluster_keys", *OUTER_KEYS,
            "--inner_cluster_keys", *INNER_KEYS,
            "--ci_method", "bca",
            "--bootstrap_samples", "10000",
            "--random_seed", "1401",
            "--output_root", str(statistics_root),
            "--formal-agent-order-contract-path",
            str(ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/formal_agent_order_contract.json"),
        )
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    write_json(
        output_root / "statistics_command_receipt.json",
        {
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"capacity-aware statistics failed: {completed.stderr}")
    new_path = statistics_root / "paired_statistics.json"
    new = load_json(new_path)
    difference = compare(old, new)
    write_json(output_root / "g14r21_old_new_capacity_comparison.json", difference)
    audit_path = output_root / "independent_recalculation.json"
    audit_command = [
        sys.executable,
        str(ROOT / "scripts/audit_g14r21_corrected_statistics.py"),
        "--corrected-statistics", str(new_path),
        "--output", str(audit_path),
    ]
    audit = subprocess.run(audit_command, cwd=ROOT, text=True, capture_output=True, check=False)
    if audit.returncode != 0:
        raise RuntimeError(f"independent capacity statistics audit failed: {audit.stderr}")
    original_verification = []
    for source in sources:
        path = Path(source["rows"]["path"])
        original_verification.append(
            {
                "path": str(path),
                "expected_sha256": source["rows"]["sha256"],
                "observed_sha256": sha256_file(path),
                "unchanged": sha256_file(path) == source["rows"]["sha256"],
            }
        )
    if not all(row["unchanged"] for row in original_verification):
        raise ValueError("immutable formal CSV protection check failed")
    write_json(
        output_root / "capacity_identity_provenance.json",
        {
            "provenance_version": "g14r22b_verified_legacy_augmentation_v1",
            "inference_from_outcomes": False,
            "inference_from_directory_names": False,
            "original_csv_modified": False,
            "sources": sources,
            "augmented_inputs": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in augmented
            ],
            "original_protection_verification": original_verification,
            "generator_repository": generator_repository,
        },
    )
    write_json(
        output_root / "analysis_summary.json",
        {
            "analysis_contract_version": "g14r22b_capacity_identity_closure_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_at": "2026-09-23",
            "literature_cutoff": "2026-09-21",
            "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
            "artifact_run_id": args.formal_root.resolve().name,
            "policy_version": "tmc_review_policy_v3_20260621",
            "evidence_level": (
                "E3_TARGETED_REPRODUCED_WITH_VERIFIED_CAPACITY_IDENTITY; "
                "bootstrap CI generated deterministically and means/effects/window sign/Holm independently recalculated"
            ),
            "git_commit": generator_repository["git_commit"],
            "git_tree": generator_repository["git_tree"],
            "generator_repository": generator_repository,
            "pair_keys": list(PAIR_KEYS),
            "outer_cluster_keys": list(OUTER_KEYS),
            "inner_cluster_keys": list(INNER_KEYS),
            "family_size": len(new["rows"]),
            "claim_counts": difference["new_claim_counts"],
            "claim_changed_count": difference["claim_changed_count"],
            "holdout_opened": False,
            "grant_signed": False,
            "token_issued": False,
            "training_runs": 0,
            "formal_rollout_runs": 0,
            "model_selection_runs": 0,
            "old_package_retained_audit_only": True,
        },
    )
    write_integrity(output_root)
    print(json.dumps({"output_root": str(output_root), "claim_counts": difference["new_claim_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
