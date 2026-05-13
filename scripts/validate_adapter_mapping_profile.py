"""Validate Alibaba adapter assignment profiles against the sample catalog."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder


OUTPUT_DIR = Path("artifacts/analysis/adapter_catalog_mapping_alignment_round8")
DEFAULT_WORKFLOW_CSV = Path("data/raw/workflow/alibaba2018/batch_task.csv")
DEFAULT_CATALOG = Path("src/data/model_catalog/sample_model_catalog.json")
PROFILES = ["legacy_batch_type", "semantic_ai_service"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow_csv_path", type=Path, default=DEFAULT_WORKFLOW_CSV)
    parser.add_argument("--adapter_catalog_path", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--workflow_ids", nargs="+", default=["j_3", "j_8"])
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _critical_path_len(node_ids: list[str], edges: list[tuple[str, str]]) -> int:
    if not node_ids:
        return 0
    parents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for src, dst in edges:
        if src in indegree and dst in indegree:
            parents[dst].append(src)
            children[src].append(dst)
            indegree[dst] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for child_id in sorted(children.get(node_id, [])):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
    longest = {node_id: 1 for node_id in node_ids}
    for node_id in order:
        parent_lengths = [longest[parent_id] for parent_id in parents.get(node_id, [])]
        if parent_lengths:
            longest[node_id] = max(parent_lengths) + 1
    return max(longest.values()) if longest else 0


def _load_catalog(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    cached_adapter_ids = {
        str(adapter_id)
        for profile in payload.get("rsu_adapter_caches", [])
        for adapter_id in profile.get("cached_adapter_ids", [])
    }
    object_adapter_ids = {str(item.get("adapter_id")) for item in payload.get("cache_objects", []) if item.get("adapter_id")}
    bundle_adapter_ids = {
        str(item.get("adapter_id"))
        for item in payload.get("adapter_state_bundles", [])
        if item.get("adapter_id")
    }
    base_model_ids = {
        str(item.get("base_model_id"))
        for item in payload.get("vehicle_base_models", [])
        if item.get("base_model_id")
    }
    adapter_sizes = {
        str(item.get("adapter_id")): float(item.get("size_mb", 0.0) or 0.0)
        for item in payload.get("cache_objects", [])
        if item.get("adapter_id")
    }
    return {
        "adapter_ids": sorted(cached_adapter_ids | object_adapter_ids | bundle_adapter_ids),
        "base_model_ids": sorted(base_model_ids),
        "adapter_sizes": adapter_sizes,
        "source_path": str(path),
    }


def _select_workflows(samples: list[dict[str, Any]], workflow_ids: list[str]) -> list[dict[str, Any]]:
    id_set = set(workflow_ids)
    return [sample for sample in samples if str(sample.get("workflow_id")) in id_set]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(
    *,
    workflow_csv_path: Path,
    workflow_ids: list[str],
    profile: str,
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = WorkflowDatasetBuilder().build_alibaba_samples(
        csv_path=workflow_csv_path,
        limit_jobs=max(32, len(workflow_ids) * 16),
        min_tasks=5,
        max_tasks=20,
        adapter_assignment_profile=profile,
    )
    selected = _select_workflows(samples, workflow_ids)
    validation_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    catalog_adapters = set(catalog.get("adapter_ids", []))
    catalog_base_models = set(catalog.get("base_model_ids", []))
    for sample in selected:
        nodes = sample.get("nodes", [])
        edges = [tuple(edge) for edge in sample.get("edges", [])]
        node_ids = [str(node.get("node_id")) for node in nodes]
        required_adapters = sorted({str(node.get("required_adapter")) for node in nodes if node.get("required_adapter")})
        required_base_models = sorted({str(node.get("required_base_model")) for node in nodes if node.get("required_base_model")})
        missing_adapters = sorted(set(required_adapters) - catalog_adapters)
        missing_base_models = sorted(set(required_base_models) - catalog_base_models)
        validation_rows.append(
            {
                "workflow_id": sample.get("workflow_id"),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "critical_path_len": _critical_path_len(node_ids, edges),
                "required_adapter_count": len(required_adapters),
                "required_adapter_ids": ";".join(required_adapters) if required_adapters else "missing",
                "required_base_model_count": len(required_base_models),
                "required_base_model_ids": ";".join(required_base_models) if required_base_models else "missing",
                "adapter_assignment_profile": profile,
                "profile_note": sample.get("adapter_assignment_profile_note", "missing"),
                "semantic_profile_multi_adapter": bool(profile == "semantic_ai_service" and len(required_adapters) >= 3),
                "all_required_adapters_in_catalog": not missing_adapters,
                "missing_adapter_ids": ";".join(missing_adapters) if missing_adapters else "none",
            }
        )
        for adapter_id in required_adapters:
            alignment_rows.append(
                {
                    "workflow_id": sample.get("workflow_id"),
                    "adapter_assignment_profile": profile,
                    "adapter_id": adapter_id,
                    "catalog_source": catalog.get("source_path", "missing"),
                    "adapter_in_catalog": adapter_id in catalog_adapters,
                    "adapter_size_mb": catalog.get("adapter_sizes", {}).get(adapter_id, "missing"),
                    "source_type": "controlled_assignment_validation" if profile == "semantic_ai_service" else "legacy_assignment_validation",
                    "missing_base_model_ids": "not_applicable",
                }
            )
        for base_model_id in required_base_models:
            alignment_rows.append(
                {
                    "workflow_id": sample.get("workflow_id"),
                    "adapter_assignment_profile": profile,
                    "adapter_id": f"base_model:{base_model_id}",
                    "catalog_source": catalog.get("source_path", "missing"),
                    "adapter_in_catalog": base_model_id in catalog_base_models,
                    "adapter_size_mb": "base_model_entry",
                    "source_type": "base_model_alignment_validation",
                    "missing_base_model_ids": ";".join(missing_base_models) if missing_base_models else "none",
                }
            )
    return validation_rows, alignment_rows


def _cache_capacity_probe() -> dict[str, Any]:
    env_path = Path("src/envs/core/vec_workflow_core_env.py")
    text = env_path.read_text(encoding="utf-8", errors="replace") if env_path.exists() else ""
    has_capacity = bool(re.search(r"cache_capacity|capacity_limit|max_cached", text))
    has_eviction = bool(re.search(r"evict|eviction|remove\(|pop\(", text, flags=re.IGNORECASE))
    return {
        "cache_capacity_source": str(env_path) if env_path.exists() else "missing",
        "cache_capacity_known": has_capacity,
        "eviction_mechanism_detected": has_eviction and has_capacity,
        "proposal_only": True,
        "cache_pressure_profile.enabled": False,
        "cache_pressure_profile.profile_name": "multi_adapter_capacity_stress",
        "cache_pressure_profile.rsu_cache_capacity_adapter_slots": 2,
        "cache_pressure_profile.eviction_policy": "lru_or_lowest_predicted_value_proposal",
        "cache_pressure_profile.admission_policy": "controlled_by_policy",
        "cache_pressure_profile.count_eviction_telemetry": True,
        "note": "No environment behavior is changed by this validation script.",
    }


def main() -> None:
    args = parse_args()
    catalog = _load_catalog(args.adapter_catalog_path)
    validation_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        profile_rows, profile_alignment = _sample_rows(
            workflow_csv_path=args.workflow_csv_path,
            workflow_ids=args.workflow_ids,
            profile=profile,
            catalog=catalog,
        )
        validation_rows.extend(profile_rows)
        alignment_rows.extend(profile_alignment)
    cache_probe = _cache_capacity_probe()
    cache_rows = [
        {
            "proposal_name": "multi_adapter_hard_joint_proposal",
            "source_file": "configs/benchmark/multi_adapter_hard_joint_proposal.yaml",
            **cache_probe,
        }
    ]
    semantic_rows = [row for row in validation_rows if row["adapter_assignment_profile"] == "semantic_ai_service"]
    semantic_multi = bool(semantic_rows and all(bool(row["semantic_profile_multi_adapter"]) for row in semantic_rows))
    semantic_aligned = bool(semantic_rows and all(bool(row["all_required_adapters_in_catalog"]) for row in semantic_rows))
    diagnosis = {
        "task_name": "adapter_catalog_mapping_alignment_round8",
        "changed_files": [
            "src/data/workflow/alibaba_dag_parser.py",
            "src/data/workflow/workflow_dataset_builder.py",
            "scripts/validate_adapter_mapping_profile.py",
            "configs/benchmark/multi_adapter_hard_joint_proposal.yaml",
            "docs/agent/adapter_catalog_mapping_alignment_round8_report.md",
        ],
        "changed_behavior": False,
        "training_run": False,
        "workflow_csv_path": str(args.workflow_csv_path),
        "adapter_catalog_path": str(args.adapter_catalog_path),
        "profiles_validated": PROFILES,
        "workflow_ids": args.workflow_ids,
        "semantic_ai_service_produces_multiple_adapters": semantic_multi,
        "semantic_ai_service_all_adapters_in_catalog": semantic_aligned,
        "semantic_ai_service_adapter_ids": sorted(
            {
                adapter_id
                for row in semantic_rows
                for adapter_id in str(row["required_adapter_ids"]).split(";")
                if adapter_id and adapter_id != "missing"
            }
        ),
        "legacy_missing_adapter_ids": sorted(
            {
                adapter_id
                for row in validation_rows
                if row["adapter_assignment_profile"] == "legacy_batch_type"
                for adapter_id in str(row["missing_adapter_ids"]).split(";")
                if adapter_id and adapter_id != "none"
            }
        ),
        "cache_capacity_eviction_remains_proposal_only": True,
        "cache_capacity_known": cache_probe["cache_capacity_known"],
        "eviction_mechanism_detected": cache_probe["eviction_mechanism_detected"],
        "generated_artifacts": {
            "adapter_mapping_profile_validation": str(args.output_dir / "adapter_mapping_profile_validation.csv"),
            "adapter_catalog_alignment_check": str(args.output_dir / "adapter_catalog_alignment_check.csv"),
            "cache_capacity_stress_proposal": str(args.output_dir / "cache_capacity_stress_proposal.csv"),
            "diagnosis_summary": str(args.output_dir / "diagnosis_summary.json"),
            "proposal_config": "configs/benchmark/multi_adapter_hard_joint_proposal.yaml",
            "report": "docs/agent/adapter_catalog_mapping_alignment_round8_report.md",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "adapter_mapping_profile_validation.csv", validation_rows)
    _write_csv(args.output_dir / "adapter_catalog_alignment_check.csv", alignment_rows)
    _write_csv(args.output_dir / "cache_capacity_stress_proposal.csv", cache_rows)
    (args.output_dir / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("adapter mapping profile validation complete")
    print(f"semantic_ai_service_produces_multiple_adapters: {semantic_multi}")
    print(f"semantic_ai_service_all_adapters_in_catalog: {semantic_aligned}")
    print(f"cache_capacity_eviction_remains_proposal_only: {diagnosis['cache_capacity_eviction_remains_proposal_only']}")
    for key, path in diagnosis["generated_artifacts"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
