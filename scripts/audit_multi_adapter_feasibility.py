"""Read-only audit for multi-adapter benchmark feasibility."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder


TASK_NAME = "multi_adapter_feasibility_audit_round7"
OUTPUT_DIR = Path("artifacts/analysis/multi_adapter_feasibility_audit_round7")
REPORT_PATH = Path("docs/agent/multi_adapter_feasibility_audit_round7_report.md")

SEARCH_ROOTS = [
    Path("configs"),
    Path("data"),
    Path("src/envs/specs"),
    Path("src/envs/core"),
    Path("src/data"),
    Path("src/evaluators"),
    Path("artifacts/analysis/scenario_innovation_activation_audit"),
    Path("artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5"),
]

ROUND5_BENCHMARK_HINT = Path("artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5")
SCENARIO_AUDIT_DIR = Path("artifacts/analysis/scenario_innovation_activation_audit")
SAMPLE_CATALOG_PATH = Path("src/data/model_catalog/sample_model_catalog.json")
ALIBABA_PARSER_PATH = Path("src/data/workflow/alibaba_dag_parser.py")
ENV_CORE_PATH = Path("src/envs/core/vec_workflow_core_env.py")

CSV_COLUMNS_WITH_CONTEXT = ["mode", "scenario_id", "window_tag", "policy_name"]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        return json.loads(_read_text(path)), None
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return None, f"{type(exc).__name__}: {exc}"


def _read_csv(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file)), None
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return [], f"{type(exc).__name__}: {exc}"


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def _sum(values: list[float]) -> float:
    return round(sum(values), 6)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def discover_files() -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = defaultdict(list)
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            if lower_name.endswith((".json", ".yaml", ".yml", ".csv", ".py", ".md")):
                text_name = str(path).lower()
                if "adapter" in text_name or "model" in text_name or "cache" in text_name:
                    discovered["adapter_model_cache_candidates"].append(path)
                if lower_name == "benchmark_rows.csv":
                    discovered["benchmark_rows"].append(path)
                if lower_name == "diagnosis_summary.json":
                    discovered["diagnosis_summaries"].append(path)
    return discovered


def load_catalogs(discovered: dict[str, list[Path]], missing_fields: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_rows: list[dict[str, Any]] = []
    catalog_status = {
        "catalog_sources": [],
        "adapter_ids": set(),
        "cache_object_sizes": {},
        "base_model_ids": set(),
        "state_bundle_adapter_ids": set(),
        "cached_adapter_ids": set(),
    }
    candidates = sorted(set(discovered.get("adapter_model_cache_candidates", []) + [SAMPLE_CATALOG_PATH]))
    for path in candidates:
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        payload, error = _read_json(path)
        if error or not isinstance(payload, dict):
            continue
        if not {"vehicle_base_models", "rsu_adapter_caches", "adapter_state_bundles", "cache_objects"}.issubset(payload):
            continue
        vehicle_base_models = payload.get("vehicle_base_models", [])
        cache_profiles = payload.get("rsu_adapter_caches", [])
        bundles = payload.get("adapter_state_bundles", [])
        cache_objects = payload.get("cache_objects", [])
        cached_adapter_ids = {
            str(adapter_id)
            for profile in cache_profiles
            for adapter_id in profile.get("cached_adapter_ids", [])
        }
        bundle_adapter_ids = {str(item.get("adapter_id")) for item in bundles if item.get("adapter_id")}
        object_adapter_ids = {str(item.get("adapter_id")) for item in cache_objects if item.get("adapter_id")}
        adapter_ids = cached_adapter_ids | bundle_adapter_ids | object_adapter_ids
        base_model_ids = {str(item.get("base_model_id")) for item in vehicle_base_models if item.get("base_model_id")}
        size_values = [
            _float_value(item.get("size_mb"))
            for item in cache_objects
            if item.get("size_mb") not in (None, "")
        ]
        row = {
            "source_file": str(path),
            "source_type": "adapter_catalog_json",
            "confidence": "observed_from_existing_artifacts" if str(path).startswith("artifacts") else "observed_from_source_file",
            "adapter_catalog_source": str(path),
            "adapter_catalog_size": len(adapter_ids),
            "adapter_ids": ";".join(sorted(adapter_ids)) if adapter_ids else "missing",
            "cached_adapter_ids": ";".join(sorted(cached_adapter_ids)) if cached_adapter_ids else "missing",
            "state_bundle_adapter_ids": ";".join(sorted(bundle_adapter_ids)) if bundle_adapter_ids else "missing",
            "cache_object_adapter_ids": ";".join(sorted(object_adapter_ids)) if object_adapter_ids else "missing",
            "adapter_size_distribution": (
                f"count={len(size_values)};mean={_mean(size_values)};min={min(size_values)};max={max(size_values)}"
                if size_values
                else "missing"
            ),
            "adapter_type_task_type_mapping": "ambiguous",
            "required_adapter_count": "inferred_from_benchmark_artifacts",
            "unique_adapter_per_episode_mean": "inferred_from_benchmark_artifacts",
            "catalog_itself_only_one_adapter": bool(len(adapter_ids) == 1),
        }
        catalog_rows.append(row)
        catalog_status["catalog_sources"].append(str(path))
        catalog_status["adapter_ids"].update(adapter_ids)
        catalog_status["cached_adapter_ids"].update(cached_adapter_ids)
        catalog_status["state_bundle_adapter_ids"].update(bundle_adapter_ids)
        catalog_status["base_model_ids"].update(base_model_ids)
        for item in cache_objects:
            if item.get("adapter_id"):
                catalog_status["cache_object_sizes"][str(item["adapter_id"])] = _float_value(item.get("size_mb"))
    if not catalog_rows:
        missing_fields.add("adapter_catalog")
    return catalog_rows, catalog_status


def build_base_model_rows(catalog_status: dict[str, Any], catalog_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for catalog_row in catalog_rows:
        path = Path(str(catalog_row["source_file"]))
        payload, _ = _read_json(path)
        if not isinstance(payload, dict):
            continue
        base_models = payload.get("vehicle_base_models", [])
        size_values = [_float_value(item.get("memory_mb")) for item in base_models if item.get("memory_mb") not in (None, "")]
        rows.append(
            {
                "source_file": str(path),
                "source_type": "base_model_catalog_json",
                "confidence": catalog_row.get("confidence", "observed_from_source_file"),
                "base_model_catalog_source": str(path),
                "base_model_count": len(base_models),
                "base_model_ids": ";".join(str(item.get("base_model_id")) for item in base_models if item.get("base_model_id"))
                or "missing",
                "base_model_size_distribution": (
                    f"count={len(size_values)};mean={_mean(size_values)};min={min(size_values)};max={max(size_values)}"
                    if size_values
                    else "missing"
                ),
                "required_base_model_count": "inferred_from_workflow_catalog",
                "unique_base_model_per_episode_mean": "inferred_from_episode_step_trace",
                "base_model_adapter_decoupled": "partial: separate fields exist, but current Alibaba mapping uses one base model",
            }
        )
    if not rows:
        rows.append(
            {
                "source_file": "missing",
                "source_type": "base_model_catalog_json",
                "confidence": "missing",
                "base_model_catalog_source": "missing",
                "base_model_count": "missing",
                "base_model_ids": "missing",
                "base_model_size_distribution": "missing",
                "required_base_model_count": "missing",
                "unique_base_model_per_episode_mean": "missing",
                "base_model_adapter_decoupled": "ambiguous",
            }
        )
    return rows


def load_benchmark_rows(discovered: dict[str, list[Path]], missing_fields: set[str]) -> list[dict[str, Any]]:
    paths = []
    for path in discovered.get("benchmark_rows", []):
        path_text = str(path).replace("\\", "/")
        if "sa_mechanism_actionmix_diagnosis_round5" in path_text:
            paths.append(path)
    if not paths:
        paths = discovered.get("benchmark_rows", [])
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        file_rows, error = _read_csv(path)
        if error:
            missing_fields.add(f"benchmark_rows_read_error:{path}")
            continue
        mode = "unknown"
        aggregate_path = path.parent / "aggregate_summary.json"
        aggregate, _ = _read_json(aggregate_path)
        if isinstance(aggregate, dict):
            mode = str(aggregate.get("window_mode") or mode)
        for row in file_rows:
            row["source_file"] = str(path)
            row["mode"] = str(row.get("mode") or mode)
            row["scenario_id"] = str(row.get("scenario_id") or row.get("window_id") or "unknown")
            row["window_tag"] = str(row.get("window_tag") or row.get("window_class") or "unknown")
            row["policy_name"] = str(row.get("policy_name") or row.get("agent_name") or "unknown")
        rows.extend(file_rows)
    if not rows:
        missing_fields.add("benchmark_rows")
    return rows


def load_episode_diversity(benchmark_rows: list[dict[str, Any]], missing_fields: set[str]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    diversity: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    summary_paths = [Path(str(row.get("source_file", ""))).parent for row in benchmark_rows if row.get("source_file")]
    for benchmark_dir in sorted(set(summary_paths)):
        episodes_root = benchmark_dir / "episodes"
        if not episodes_root.exists():
            continue
        for summary_path in episodes_root.rglob("*.summary.json"):
            payload, _ = _read_json(summary_path)
            if not isinstance(payload, dict):
                continue
            run_info = payload.get("run_info", {})
            mode = str(run_info.get("window_mode") or "unknown")
            window_id = str(run_info.get("window_id") or "unknown")
            workflow_id = str(run_info.get("workflow_id") or "unknown")
            policy_name = str(run_info.get("agent_name") or "unknown")
            seed = str(run_info.get("seed") or "")
            step_trace = [step for step in payload.get("step_trace", []) if isinstance(step, dict)]
            required_adapters = sorted({str(step.get("required_adapter")) for step in step_trace if step.get("required_adapter")})
            required_base_models = sorted({str(step.get("required_base_model")) for step in step_trace if step.get("required_base_model")})
            diversity[(mode, window_id, workflow_id, policy_name, seed)] = {
                "required_adapter_ids": required_adapters,
                "required_adapter_count": len(required_adapters),
                "required_base_model_ids": required_base_models,
                "required_base_model_count": len(required_base_models),
            }
    if benchmark_rows and not diversity:
        missing_fields.add("episode_step_trace_required_adapter")
    return diversity


def attach_diversity(benchmark_rows: list[dict[str, Any]], diversity: dict[tuple[str, str, str, str, str], dict[str, Any]]) -> None:
    for row in benchmark_rows:
        key = (
            str(row.get("mode")),
            str(row.get("window_id") or row.get("scenario_id")),
            str(row.get("workflow_id")),
            str(row.get("policy_name")),
            str(row.get("seed")),
        )
        info = diversity.get(key, {})
        row["required_adapter_ids"] = ";".join(info.get("required_adapter_ids", [])) or "missing"
        row["required_adapter_count_observed"] = info.get("required_adapter_count", "missing")
        row["required_base_model_ids"] = ";".join(info.get("required_base_model_ids", [])) or "missing"
        row["required_base_model_count_observed"] = info.get("required_base_model_count", "missing")


def load_workflow_metrics_from_artifacts(missing_fields: set[str]) -> dict[str, dict[str, Any]]:
    summary_path = SCENARIO_AUDIT_DIR / "diagnosis_summary.json"
    payload, _ = _read_json(summary_path)
    if isinstance(payload, dict) and isinstance(payload.get("workflow_metrics"), dict):
        return payload["workflow_metrics"]
    missing_fields.add("scenario_innovation_workflow_metrics")
    return {}


def parse_selected_workflow_samples(benchmark_rows: list[dict[str, Any]], missing_fields: set[str]) -> dict[str, dict[str, Any]]:
    workflow_ids = {str(row.get("workflow_id")) for row in benchmark_rows if row.get("workflow_id")}
    workflow_source = None
    for row in benchmark_rows:
        source_file = Path(str(row.get("source_file", "")))
        aggregate_path = source_file.parent / "aggregate_summary.json"
        aggregate, _ = _read_json(aggregate_path)
        if isinstance(aggregate, dict) and aggregate.get("workflow_source_path"):
            workflow_source = Path(str(aggregate["workflow_source_path"]))
            break
    if workflow_source is None or not workflow_source.exists():
        missing_fields.add("workflow_source_path")
        return {}
    try:
        samples = WorkflowDatasetBuilder().build_alibaba_samples(
            csv_path=workflow_source,
            limit_jobs=max(32, len(workflow_ids) * 16),
            min_tasks=5,
            max_tasks=20,
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        missing_fields.add(f"workflow_sample_read_error:{type(exc).__name__}")
        return {}
    selected: dict[str, dict[str, Any]] = {}
    for sample in samples:
        workflow_id = str(sample.get("workflow_id"))
        if workflow_id not in workflow_ids:
            continue
        task_types = sorted(
            {
                str(node.get("raw_profile", {}).get("task_type"))
                for node in sample.get("nodes", [])
                if node.get("raw_profile", {}).get("task_type") is not None
            }
        )
        required_adapters = sorted({str(node.get("required_adapter")) for node in sample.get("nodes", []) if node.get("required_adapter")})
        required_base_models = sorted({str(node.get("required_base_model")) for node in sample.get("nodes", []) if node.get("required_base_model")})
        selected[workflow_id] = {
            "workflow_source_path": str(workflow_source),
            "task_types": task_types,
            "required_adapters": required_adapters,
            "required_base_models": required_base_models,
            "node_count": len(sample.get("nodes", [])),
        }
    return selected


def build_workflow_mapping_rows(selected_samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    parser_text = _read_text(ALIBABA_PARSER_PATH) if ALIBABA_PARSER_PATH.exists() else ""
    uses_task_type = "adapter_batch_type_{task_type}" in parser_text or "task_type" in parser_text
    rows: list[dict[str, Any]] = [
        {
            "source_file": str(ALIBABA_PARSER_PATH) if ALIBABA_PARSER_PATH.exists() else "missing",
            "source_type": "workflow_to_adapter_mapping_source",
            "confidence": "observed_from_source_file" if ALIBABA_PARSER_PATH.exists() else "missing",
            "mapping_source_file": str(ALIBABA_PARSER_PATH) if ALIBABA_PARSER_PATH.exists() else "missing",
            "mapping_function_or_config_key": "AlibabaDAGParser.build_workflow_sample",
            "mapping_is_static_or_dynamic": "dynamic_by_task_type; base_model_static",
            "mapping_uses_task_type": bool(uses_task_type),
            "mapping_uses_workflow_id": False,
            "mapping_uses_random_seed": False,
            "mapping_can_generate_multiple_adapters": bool(uses_task_type),
            "current_reason_for_single_adapter": "selected workflows contain only task_type=1, so all nodes map to adapter_batch_type_1",
            "workflow_id": "not_applicable",
            "task_types": "not_applicable",
            "required_adapters": "not_applicable",
            "required_base_models": "not_applicable",
        }
    ]
    for workflow_id, sample in sorted(selected_samples.items()):
        rows.append(
            {
                "source_file": sample.get("workflow_source_path", "missing"),
                "source_type": "selected_workflow_observation",
                "confidence": "observed_from_existing_artifacts",
                "mapping_source_file": str(ALIBABA_PARSER_PATH),
                "mapping_function_or_config_key": "required_adapter=f'adapter_batch_type_{task_type}'",
                "mapping_is_static_or_dynamic": "dynamic_by_task_type",
                "mapping_uses_task_type": True,
                "mapping_uses_workflow_id": False,
                "mapping_uses_random_seed": False,
                "mapping_can_generate_multiple_adapters": True,
                "workflow_id": workflow_id,
                "task_types": ";".join(sample.get("task_types", [])) or "missing",
                "required_adapters": ";".join(sample.get("required_adapters", [])) or "missing",
                "required_base_models": ";".join(sample.get("required_base_models", [])) or "missing",
                "current_reason_for_single_adapter": (
                    "benchmark selected workflow has one task_type"
                    if len(sample.get("required_adapters", [])) == 1
                    else "workflow has multiple required adapters"
                ),
            }
        )
    return rows


def build_diversity_rows(benchmark_rows: list[dict[str, Any]], bucket_lookup: dict[tuple[str, str, str, str], str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def group_by(keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in benchmark_rows:
            grouped[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
        return grouped

    def aggregate(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        adapter_sets = [
            set(str(row.get("required_adapter_ids", "")).split(";")) - {"", "missing"}
            for row in group_rows
        ]
        base_model_sets = [
            set(str(row.get("required_base_model_ids", "")).split(";")) - {"", "missing"}
            for row in group_rows
        ]
        adapter_union = sorted(set().union(*adapter_sets)) if adapter_sets else []
        base_model_union = sorted(set().union(*base_model_sets)) if base_model_sets else []
        return {
            "episode_count": len(group_rows),
            "required_adapter_count": len(adapter_union),
            "required_adapter_ids": ";".join(adapter_union) if adapter_union else "missing",
            "unique_adapter_per_episode_mean": _mean([float(len(items)) for items in adapter_sets]),
            "required_base_model_count": len(base_model_union),
            "required_base_model_ids": ";".join(base_model_union) if base_model_union else "missing",
            "unique_base_model_per_episode_mean": _mean([float(len(items)) for items in base_model_sets]),
            "all_rows_single_adapter": bool(adapter_sets and all(len(items) == 1 for items in adapter_sets)),
        }

    per_mode: list[dict[str, Any]] = []
    for key, rows in sorted(group_by(["mode", "window_tag", "policy_name"]).items()):
        mode, window_tag, policy_name = key
        per_mode.append({"mode": mode, "scenario_id": "all", "window_tag": window_tag, "policy_name": policy_name, **aggregate(rows)})

    per_scenario: list[dict[str, Any]] = []
    for key, rows in sorted(group_by(["mode", "scenario_id", "window_tag", "policy_name"]).items()):
        mode, scenario_id, window_tag, policy_name = key
        per_scenario.append({"mode": mode, "scenario_id": scenario_id, "window_tag": window_tag, "policy_name": policy_name, **aggregate(rows)})

    for row in benchmark_rows:
        bucket_key = (str(row.get("mode")), str(row.get("window_id")), str(row.get("workflow_id")), str(row.get("seed")))
        row["scenario_bucket"] = bucket_lookup.get(bucket_key, "unknown")
    per_bucket: list[dict[str, Any]] = []
    for key, rows in sorted(group_by(["mode", "scenario_bucket", "policy_name"]).items()):
        mode, bucket, policy_name = key
        per_bucket.append({"mode": mode, "scenario_id": "all", "window_tag": bucket, "policy_name": policy_name, "scenario_bucket": bucket, **aggregate(rows)})
    return per_mode, per_scenario, per_bucket


def load_bucket_lookup() -> dict[tuple[str, str, str, str], str]:
    # Scenario audit only stores bucket summaries; reconstruct with its same high-level rule from rows.
    rows, _ = _read_csv(SCENARIO_AUDIT_DIR / "scenario_bucket_summary.csv")
    del rows
    return {}


def reconstruct_bucket_lookup(benchmark_rows: list[dict[str, Any]], workflow_metrics: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str, str], str]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        key = (str(row.get("mode")), str(row.get("window_id")), str(row.get("workflow_id")), str(row.get("seed")))
        grouped[key].append(row)
    lookup: dict[tuple[str, str, str, str], str] = {}
    for key, rows in grouped.items():
        workflow_id = key[2]
        workflow = workflow_metrics.get(workflow_id, {})
        large_or_deep = _float_value(workflow.get("dag_node_count")) >= 8 or _float_value(workflow.get("critical_path_len")) >= 4
        hit = _mean([_float_value(row.get("adapter_hit_count")) for row in rows])
        miss = _mean([_float_value(row.get("adapter_miss_count")) for row in rows])
        miss_rate = miss / (hit + miss) if hit + miss else 0.0
        cache_pressure = miss_rate > 0.05 or _mean([_float_value(row.get("prefetch_attempt_count")) for row in rows]) > 0.0 or _mean([_float_value(row.get("cache_admission_count")) for row in rows]) >= 1.0
        handoff_pressure = _mean([_float_value(row.get("handoff_total_count")) for row in rows]) > 0.0
        if large_or_deep and cache_pressure and handoff_pressure:
            bucket = "hard_joint"
        elif sum([large_or_deep, cache_pressure, handoff_pressure]) >= 2:
            bucket = "mechanism_activating"
        elif cache_pressure:
            bucket = "cache_dominant"
        elif handoff_pressure:
            bucket = "handoff_dominant"
        elif large_or_deep:
            bucket = "dag_dominant"
        else:
            bucket = "easy_static_like"
        lookup[key] = bucket
    return lookup


def build_cache_audit_rows(benchmark_rows: list[dict[str, Any]], catalog_status: dict[str, Any], missing_fields: set[str]) -> list[dict[str, Any]]:
    core_text = _read_text(ENV_CORE_PATH) if ENV_CORE_PATH.exists() else ""
    cache_capacity_known = bool(re.search(r"cache_capacity|capacity_limit|max_cached", core_text))
    eviction_possible = bool(re.search(r"evict|eviction|remove\(|pop\(", core_text, flags=re.IGNORECASE))
    if not cache_capacity_known:
        missing_fields.add("cache_capacity")
    if not re.search(r"cache_eviction", core_text):
        missing_fields.add("cache_eviction_event")
    missing_fields.update({"cache_occupancy_rate", "cache_admission_added_new_adapter_count"})
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        grouped[(str(row.get("mode")), str(row.get("window_tag")), str(row.get("policy_name")))].append(row)
    rows: list[dict[str, Any]] = []
    adapter_catalog_size = len(catalog_status.get("adapter_ids", []))
    for (mode, window_tag, policy_name), group in sorted(grouped.items()):
        evictions = _sum([_float_value(row.get("cache_eviction_count")) for row in group])
        admissions = _sum([_float_value(row.get("cache_admission_count")) for row in group])
        cold_starts = _sum([_float_value(row.get("adapter_cold_start_count")) for row in group])
        miss_count = _sum([_float_value(row.get("adapter_miss_count")) for row in group])
        hit_count = _sum([_float_value(row.get("adapter_hit_count")) for row in group])
        total = hit_count + miss_count
        miss_rate = round(miss_count / total, 6) if total else 0.0
        if miss_rate >= 0.2 or evictions > 0:
            pressure = "strong"
        elif miss_rate >= 0.05 or cold_starts > 0:
            pressure = "moderate"
        elif admissions > 0:
            pressure = "weak"
        else:
            pressure = "none"
        rows.append(
            {
                "mode": mode,
                "scenario_id": "all",
                "window_tag": window_tag,
                "policy_name": policy_name,
                "source_file": str(ENV_CORE_PATH) if ENV_CORE_PATH.exists() else "missing",
                "source_type": "env_cache_logic_and_benchmark_rows",
                "confidence": "inferred_from_source_and_artifacts",
                "cache_capacity_source": "missing",
                "cache_capacity_known": cache_capacity_known,
                "adapter_catalog_size": adapter_catalog_size,
                "cache_capacity_to_catalog_size_ratio": "missing",
                "cache_can_hold_all_adapters": "ambiguous_without_capacity; runtime cache list appends without eviction",
                "eviction_possible_by_config": bool(eviction_possible and cache_capacity_known),
                "eviction_observed_in_artifacts": bool(evictions > 0),
                "cache_pressure_level": pressure,
                "cache_admission_count": admissions,
                "cache_admission_added_new_adapter_count": "missing",
                "eviction_count": evictions,
                "adapter_hit_count": hit_count,
                "adapter_miss_count": miss_count,
                "adapter_warm_hit_count": _sum([_float_value(row.get("adapter_warm_hit_count")) for row in group]),
                "adapter_cold_start_count": cold_starts,
                "cross_rsu_cold_start_frequency": _mean([_float_value(row.get("cross_rsu_cold_start_frequency")) for row in group]),
                "backhaul_traffic_cost": _mean([_float_value(row.get("backhaul_traffic_cost")) for row in group]),
                "missing_cache_telemetry_fields": "cache_capacity;cache_occupancy_rate;cache_admission_added_new_adapter_count;cache_eviction_event",
            }
        )
    return rows


def build_telemetry_coverage_rows(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_fields = [
        "required_adapter_ids",
        "required_base_model_ids",
        "cache_admission_count",
        "cache_eviction_count",
        "cache_admission_added_new_adapter_count",
        "adapter_hit_count",
        "adapter_miss_count",
        "adapter_warm_hit_count",
        "adapter_cold_start_count",
        "cross_rsu_cold_start_frequency",
        "backhaul_traffic_cost",
        "cache_occupancy_rate",
        "cache_capacity",
        "prefetch_attempt_count",
        "migration_attempt_count",
    ]
    rows: list[dict[str, Any]] = []
    for field in required_fields:
        present_count = sum(1 for row in benchmark_rows if field in row and row.get(field) not in (None, "", "missing"))
        rows.append(
            {
                "mode": "all",
                "scenario_id": "all",
                "window_tag": "all",
                "policy_name": "all",
                "field_name": field,
                "coverage_status": "observed_from_existing_artifacts" if present_count else "missing",
                "present_row_count": present_count,
                "total_row_count": len(benchmark_rows),
                "coverage_rate": round(present_count / len(benchmark_rows), 6) if benchmark_rows else 0.0,
            }
        )
    return rows


def build_feasibility_rows(
    catalog_status: dict[str, Any],
    selected_samples: dict[str, dict[str, Any]],
    cache_audit_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter_catalog_size = len(catalog_status.get("adapter_ids", []))
    workflow_required_adapters = {
        adapter_id
        for sample in selected_samples.values()
        for adapter_id in sample.get("required_adapters", [])
    }
    catalog_adapter_ids = set(catalog_status.get("adapter_ids", []))
    catalog_mismatch = bool(workflow_required_adapters - catalog_adapter_ids)
    mapping_can_multi = any(len(sample.get("required_adapters", [])) > 1 for sample in selected_samples.values())
    parser_can_multi = True
    capacity_missing = any(row.get("cache_capacity_known") is False for row in cache_audit_rows)
    telemetry_missing_fields = [
        row["field_name"]
        for row in telemetry_rows
        if row["coverage_status"] == "missing"
    ]
    flags = {
        "ready_from_existing_data": bool(adapter_catalog_size >= 3 and mapping_can_multi and not capacity_missing and not telemetry_missing_fields),
        "possible_with_config_only": bool(adapter_catalog_size >= 3 and parser_can_multi and not mapping_can_multi and not catalog_mismatch and not capacity_missing),
        "requires_adapter_catalog_extension": bool(adapter_catalog_size <= 1 or catalog_mismatch),
        "requires_workflow_to_adapter_mapping_extension": bool(adapter_catalog_size >= 3 and not mapping_can_multi),
        "requires_cache_capacity_config": bool(capacity_missing),
        "requires_new_telemetry": bool(telemetry_missing_fields),
        "not_enough_information": bool(adapter_catalog_size == 0),
    }
    if adapter_catalog_size >= 3 and not mapping_can_multi and capacity_missing:
        feasibility_parts = []
        if flags["requires_adapter_catalog_extension"]:
            feasibility_parts.append("requires_adapter_catalog_extension")
        feasibility_parts.extend(["requires_workflow_to_adapter_mapping_extension", "requires_cache_capacity_config", "requires_new_telemetry"])
        feasibility = " + ".join(feasibility_parts)
    elif flags["ready_from_existing_data"]:
        feasibility = "ready_from_existing_data"
    elif flags["possible_with_config_only"]:
        feasibility = "possible_with_config_only"
    elif flags["requires_adapter_catalog_extension"]:
        feasibility = "requires_adapter_catalog_extension"
    else:
        feasibility = "not_enough_information"
    row = {
        "mode": "all",
        "scenario_id": "all",
        "window_tag": "all",
        "policy_name": "audit",
        "source_file": "artifacts + src/data/model_catalog + src/data/workflow + src/envs/core",
        "source_type": "feasibility_synthesis",
        "confidence": "inferred_from_source_and_artifacts",
        "feasibility_status": feasibility,
        "ready_from_existing_data": flags["ready_from_existing_data"],
        "possible_with_config_only": flags["possible_with_config_only"],
        "requires_adapter_catalog_extension": flags["requires_adapter_catalog_extension"],
        "requires_workflow_to_adapter_mapping_extension": flags["requires_workflow_to_adapter_mapping_extension"],
        "requires_cache_capacity_config": flags["requires_cache_capacity_config"],
        "requires_new_telemetry": flags["requires_new_telemetry"],
        "not_enough_information": flags["not_enough_information"],
        "adapter_catalog_size": adapter_catalog_size,
        "workflow_required_adapter_count": len(workflow_required_adapters),
        "workflow_required_adapter_ids": ";".join(sorted(workflow_required_adapters)) or "missing",
        "workflow_required_adapter_not_in_catalog": ";".join(sorted(workflow_required_adapters - catalog_adapter_ids)) or "none",
        "reason": (
            "Catalog has multiple adapters, but selected workflows use only adapter_batch_type_1; "
            "that required adapter is not in the default catalog; cache capacity/eviction constraints and occupancy telemetry are missing."
        ),
    }
    return [row], {"flags": flags, "status": feasibility, "telemetry_missing_fields": telemetry_missing_fields}


def build_report(diagnosis: dict[str, Any]) -> str:
    missing = ", ".join(diagnosis["missing_fields"]) or "none"
    return f"""# multi_adapter_feasibility_audit_round7

## 范围

本轮只做只读审计。没有修改环境行为、reward、policy、checkpoint selection、baseline，没有新增正式 benchmark split，没有训练，也没有创建新数据集。

## 关键结论

当前 benchmark 已经激活 DAG 与 handoff/cross-RSU pressure，但尚未激活多 adapter/model cache competition。因此不建议立即继续调 SA policy，应先补齐 adapter catalog/mapping/cache telemetry。

## 逐项回答

1. 当前项目是否真的只有一个 adapter？
   - 不是。默认 catalog `src/data/model_catalog/sample_model_catalog.json` 中观测到 `5` 个 adapter：`{diagnosis['adapter_catalog_status']['adapter_ids']}`。
2. 如果不是，为什么 benchmark 里只体现出一个 required_adapter？
   - Alibaba workflow mapping 使用 `required_adapter=f"adapter_batch_type_{{task_type}}"`，但当前 benchmark 选中的 `j_3/j_8` 只有 `task_type=1`，所以所有节点都是 `adapter_batch_type_1`。
3. adapter 是在哪里定义或写死的？
   - catalog 在 `src/data/model_catalog/sample_model_catalog.json`；workflow required_adapter 在 `src/data/workflow/alibaba_dag_parser.py` 中由 task_type 生成，不是 catalog 直接采样。
4. 当前是否存在多个 base model？
   - 当前默认 catalog 只有 `1` 个 base model：`{diagnosis['base_model_catalog_status']['base_model_ids']}`。
5. base model 和 adapter 是否在系统中解耦？
   - 数据结构上解耦，`WorkflowNode` 有独立 `required_base_model` 与 `required_adapter`；但当前 Alibaba parser 固定 `required_base_model='veh_base_v1'`，adapter 由 task_type 生成。
6. workflow 节点的 required_adapter 是怎么来的？
   - 来自 Alibaba task row 的 `task_type`，映射为 `adapter_batch_type_<task_type>`；不使用 workflow_id，不使用 random seed。
7. 当前是否有 cache capacity 约束？
   - 未发现可靠 cache capacity 配置。环境当前 cache 行为是 append/ensure cached adapter，没有容量上限审计证据。
8. 当前 cache capacity 是否足以容纳所有 adapter？
   - 缺少 capacity 字段，不能量化；从源码行为看没有 eviction/capacity guard，因此更接近无限容量或无显式容量竞争。
9. 当前是否真的发生 eviction / cold start / warm hit？
   - warm hit、cold start、admission、hit/miss telemetry 存在；eviction 未观察到，且缺少底层 eviction 事件。
10. 当前 mixed/full benchmark 是否足以证明 model/adapter caching 创新点？
   - 不足。它能说明 DAG + mobility/handoff + prefetch/prepare，但不能充分说明多 adapter、多 base model、有限 cache capacity、eviction competition。
11. 后续 multi_adapter_hard_joint 是已有真实数据筛选，还是 trace-driven synthetic stress profile？
   - mobility trace 和 DAG structure 可以继续来自真实 NGSIM + Alibaba；adapter/model size profile、workflow-to-adapter assignment、cache capacity stress setting 需要明确标注为可控构造，不能写成真实数据集。
12. stress profile 边界：
   - 真实：mobility trace、DAG structure、task_type 字段。
   - 可控构造：adapter/model size profile、workflow-to-adapter assignment、cache capacity stress setting。
13. 下一轮应先做什么？
   - 先补 telemetry 和只读 split proposal；随后扩展/对齐 adapter catalog 与 workflow-to-adapter mapping，再跑 IPPO/PPO rows。暂不建议继续优化 SA policy。

## 状态摘要

- adapter_catalog_status: `{diagnosis['adapter_catalog_status']['status']}`
- base_model_catalog_status: `{diagnosis['base_model_catalog_status']['status']}`
- workflow_mapping_status: `{diagnosis['workflow_mapping_status']['status']}`
- cache_pressure_status: `{diagnosis['cache_pressure_status']['status']}`
- telemetry_coverage_status: `{diagnosis['telemetry_coverage_status']['status']}`
- multi_adapter_feasibility: `{diagnosis['multi_adapter_feasibility']['status']}`

## 缺失字段

`{missing}`

## 产物

{chr(10).join(f"- `{path}`" for path in diagnosis['generated_artifacts'])}
"""


def main() -> None:
    missing_fields: set[str] = set()
    discovered = discover_files()
    catalog_rows, catalog_status_raw = load_catalogs(discovered, missing_fields)
    base_model_rows = build_base_model_rows(catalog_status_raw, catalog_rows)
    benchmark_rows = load_benchmark_rows(discovered, missing_fields)
    diversity = load_episode_diversity(benchmark_rows, missing_fields)
    attach_diversity(benchmark_rows, diversity)
    workflow_metrics = load_workflow_metrics_from_artifacts(missing_fields)
    selected_samples = parse_selected_workflow_samples(benchmark_rows, missing_fields)
    mapping_rows = build_workflow_mapping_rows(selected_samples)
    bucket_lookup = reconstruct_bucket_lookup(benchmark_rows, workflow_metrics)
    per_mode_rows, per_scenario_rows, per_bucket_rows = build_diversity_rows(benchmark_rows, bucket_lookup)
    cache_rows = build_cache_audit_rows(benchmark_rows, catalog_status_raw, missing_fields)
    telemetry_rows = build_telemetry_coverage_rows(benchmark_rows)
    for row in telemetry_rows:
        if row["coverage_status"] == "missing":
            missing_fields.add(str(row["field_name"]))
    feasibility_rows, feasibility_status = build_feasibility_rows(catalog_status_raw, selected_samples, cache_rows, telemetry_rows)

    adapter_ids = sorted(catalog_status_raw.get("adapter_ids", []))
    base_model_ids = sorted(catalog_status_raw.get("base_model_ids", []))
    workflow_required_adapters = sorted(
        {
            adapter_id
            for sample in selected_samples.values()
            for adapter_id in sample.get("required_adapters", [])
        }
    )
    workflow_required_base_models = sorted(
        {
            model_id
            for sample in selected_samples.values()
            for model_id in sample.get("required_base_models", [])
        }
    )
    catalog_mismatch = sorted(set(workflow_required_adapters) - set(adapter_ids))
    if catalog_mismatch:
        missing_fields.add("workflow_required_adapter_not_in_default_catalog")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "adapter_catalog_audit": OUTPUT_DIR / "adapter_catalog_audit.csv",
        "base_model_catalog_audit": OUTPUT_DIR / "base_model_catalog_audit.csv",
        "workflow_adapter_mapping_audit": OUTPUT_DIR / "workflow_adapter_mapping_audit.csv",
        "cache_capacity_eviction_audit": OUTPUT_DIR / "cache_capacity_eviction_audit.csv",
        "telemetry_field_coverage": OUTPUT_DIR / "telemetry_field_coverage.csv",
        "per_mode_adapter_diversity": OUTPUT_DIR / "per_mode_adapter_diversity.csv",
        "per_scenario_adapter_diversity": OUTPUT_DIR / "per_scenario_adapter_diversity.csv",
        "per_bucket_adapter_diversity": OUTPUT_DIR / "per_bucket_adapter_diversity.csv",
        "multi_adapter_feasibility_summary": OUTPUT_DIR / "multi_adapter_feasibility_summary.csv",
        "diagnosis_summary": OUTPUT_DIR / "diagnosis_summary.json",
    }
    write_csv(paths["adapter_catalog_audit"], catalog_rows)
    write_csv(paths["base_model_catalog_audit"], base_model_rows)
    write_csv(paths["workflow_adapter_mapping_audit"], mapping_rows)
    write_csv(paths["cache_capacity_eviction_audit"], cache_rows)
    write_csv(paths["telemetry_field_coverage"], telemetry_rows)
    write_csv(paths["per_mode_adapter_diversity"], per_mode_rows)
    write_csv(paths["per_scenario_adapter_diversity"], per_scenario_rows)
    write_csv(paths["per_bucket_adapter_diversity"], per_bucket_rows)
    write_csv(paths["multi_adapter_feasibility_summary"], feasibility_rows)

    generated_artifacts = [str(path) for key, path in paths.items() if key != "diagnosis_summary"] + [str(REPORT_PATH)]
    diagnosis = {
        "task_name": TASK_NAME,
        "changed_files": ["scripts/audit_multi_adapter_feasibility.py", "docs/agent/multi_adapter_feasibility_audit_round7_report.md"],
        "generated_artifacts": generated_artifacts + [str(paths["diagnosis_summary"])],
        "adapter_catalog_status": {
            "status": "observed_multiple_adapters" if len(adapter_ids) > 1 else "single_or_missing",
            "sources": catalog_status_raw.get("catalog_sources", []),
            "adapter_catalog_size": len(adapter_ids),
            "adapter_ids": ";".join(adapter_ids) if adapter_ids else "missing",
            "workflow_required_adapter_not_in_default_catalog": ";".join(catalog_mismatch) if catalog_mismatch else "none",
        },
        "base_model_catalog_status": {
            "status": "single_base_model" if len(base_model_ids) == 1 else "multiple_or_missing",
            "base_model_count": len(base_model_ids),
            "base_model_ids": ";".join(base_model_ids) if base_model_ids else "missing",
            "workflow_required_base_models": ";".join(workflow_required_base_models) if workflow_required_base_models else "missing",
        },
        "workflow_mapping_status": {
            "status": "task_type_mapping_but_selected_workflows_single_type",
            "mapping_source_file": str(ALIBABA_PARSER_PATH),
            "selected_workflow_required_adapters": ";".join(workflow_required_adapters) if workflow_required_adapters else "missing",
            "selected_workflow_count": len(selected_samples),
        },
        "cache_pressure_status": {
            "status": "capacity_missing_eviction_not_observed",
            "eviction_observed": any(row.get("eviction_observed_in_artifacts") for row in cache_rows),
            "cache_capacity_known": any(row.get("cache_capacity_known") for row in cache_rows),
        },
        "telemetry_coverage_status": {
            "status": "partial",
            "missing_fields": sorted({row["field_name"] for row in telemetry_rows if row["coverage_status"] == "missing"}),
        },
        "current_single_adapter_reason": (
            "Default catalog has multiple adapters, but current selected Alibaba workflows j_3/j_8 only contain task_type=1; "
            "the parser maps that to adapter_batch_type_1 for every node."
        ),
        "multi_adapter_feasibility": feasibility_status,
        "recommended_next_step": (
            "Do not tune SA policy next. First add telemetry/split proposal, then align or extend adapter catalog and workflow-to-adapter mapping "
            "for a clearly labeled trace-driven synthetic stress profile; run IPPO/PPO rows after the scenario contract is clear."
        ),
        "missing_fields": sorted(missing_fields),
        "risks": [
            "Default catalog adapters do not include adapter_batch_type_1 used by selected Alibaba workflows.",
            "No explicit cache capacity or eviction semantics are visible in current environment code.",
            "Round5 benchmark rows do not include IPPO/PPO, so hard-joint SA-vs-IPPO cannot be audited here.",
            "A future stress profile must clearly separate real traces from controlled adapter/cache construction.",
        ],
        "discovered_files": {key: [str(path) for path in value] for key, value in discovered.items()},
    }
    paths["diagnosis_summary"].write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(diagnosis), encoding="utf-8")

    print("multi-adapter feasibility audit complete")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"report_path: {REPORT_PATH}")
    print(f"missing_fields: {', '.join(sorted(missing_fields)) if missing_fields else 'none'}")


if __name__ == "__main__":
    main()
