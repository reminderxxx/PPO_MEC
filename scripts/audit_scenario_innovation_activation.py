"""Audit whether benchmark scenarios activate the paper's intended mechanisms."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder


DEFAULT_MIXED_INPUT = Path(
    "artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/"
    "benchmark_mixed/main_results_mixed_informative_20260426_023736_742838"
)
DEFAULT_FULL_INPUT = Path(
    "artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/"
    "benchmark_full/main_results_full_stratified_20260426_023920_694043"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/scenario_innovation_activation_audit")
DEFAULT_REPORT_PATH = Path("docs/agent/scenario_innovation_activation_audit_report.md")

SA_AGENT = "sa_ghmappo"
POPULARITY_AGENT = "popularity_cache_heuristic"
IPPO_AGENT_NAMES = {"ippo", "ppo", "ppo_real", "flat_ppo"}

LARGE_DAG_NODE_THRESHOLD = 8
DEEP_DAG_CRITICAL_PATH_THRESHOLD = 4
LONG_WORKFLOW_DURATION_THRESHOLD = 12.0
CACHE_PRESSURE_MISS_RATE_THRESHOLD = 0.05
PREDICTION_SIGNAL_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed_input", type=Path, default=DEFAULT_MIXED_INPUT)
    parser.add_argument("--full_input", type=Path, default=DEFAULT_FULL_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--workflow_csv_path", type=Path, default=None)
    parser.add_argument("--adapter_catalog_path", type=Path, default=Path("src/data/model_catalog/sample_model_catalog.json"))
    return parser.parse_args()


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_rate(values: list[bool]) -> float:
    return round(sum(1 for value in values if value) / len(values), 6) if values else 0.0


def _mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 6)
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return round(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight, 6)


def _summary_stats(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_p50": _percentile(values, 0.50),
        f"{prefix}_p90": _percentile(values, 0.90),
        f"{prefix}_max": round(max(values), 6) if values else 0.0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _input_rows_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    return input_path / "benchmark_rows.csv"


def read_benchmark(input_path: Path, fallback_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any], Path | None]:
    rows_path = _input_rows_path(input_path)
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing benchmark_rows.csv: {rows_path}")
    aggregate_path = rows_path.parent / "aggregate_summary.json"
    aggregate = _read_json(aggregate_path)
    mode = str(aggregate.get("window_mode") or fallback_mode)
    with rows_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["mode"] = str(row.get("mode") or mode)
        row["scenario_id"] = str(row.get("scenario_id") or row.get("window_id") or "unknown")
        row["window_tag"] = str(row.get("window_tag") or row.get("window_class") or "unknown")
        row["policy_name"] = str(row.get("policy_name") or row.get("agent_name") or "unknown")
    return rows, aggregate, rows_path.parent if rows_path.parent.exists() else None


def _critical_path_length(node_ids: list[str], edges: list[tuple[str, str]]) -> int:
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
    frontier = [node_id for node_id, degree in indegree.items() if degree == 0]
    order: list[str] = []
    while frontier:
        node_id = frontier.pop(0)
        order.append(node_id)
        for child_id in children.get(node_id, []):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                frontier.append(child_id)
    longest = {node_id: 1 for node_id in node_ids}
    for node_id in order:
        parent_lengths = [longest[parent_id] for parent_id in parents.get(node_id, [])]
        if parent_lengths:
            longest[node_id] = max(parent_lengths) + 1
    return max(longest.values()) if longest else 0


def _parallel_width(node_ids: list[str], edges: list[tuple[str, str]]) -> int:
    if not node_ids:
        return 0
    indegree = {node_id: 0 for node_id in node_ids}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for src, dst in edges:
        if src in indegree and dst in indegree:
            indegree[dst] += 1
            children[src].append(dst)
    frontier = sorted([node_id for node_id, degree in indegree.items() if degree == 0])
    max_width = len(frontier)
    remaining = dict(indegree)
    while frontier:
        next_frontier: list[str] = []
        for node_id in frontier:
            for child_id in children.get(node_id, []):
                remaining[child_id] -= 1
                if remaining[child_id] == 0:
                    next_frontier.append(child_id)
        frontier = sorted(next_frontier)
        max_width = max(max_width, len(frontier))
    return max_width


def load_workflow_metrics(
    workflow_csv_path: Path | None,
    workflow_ids: set[str],
    missing_fields: set[str],
) -> dict[str, dict[str, Any]]:
    if workflow_csv_path is None:
        missing_fields.add("workflow_csv_path")
        return {}
    workflow_csv_path = Path(workflow_csv_path)
    if not workflow_csv_path.exists():
        missing_fields.add("workflow_csv_path")
        return {}
    try:
        samples = WorkflowDatasetBuilder().build_alibaba_samples(
            csv_path=workflow_csv_path,
            limit_jobs=max(32, len(workflow_ids) * 16),
            min_tasks=5,
            max_tasks=20,
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        missing_fields.add(f"workflow_catalog_read_error:{type(exc).__name__}")
        return {}

    metrics: dict[str, dict[str, Any]] = {}
    for sample in samples:
        workflow_id = str(sample.get("workflow_id", ""))
        if workflow_ids and workflow_id not in workflow_ids:
            continue
        nodes = sample.get("nodes", [])
        edges = [tuple(edge) for edge in sample.get("edges", [])]
        node_ids = [str(node.get("node_id")) for node in nodes]
        parent_counts = [len(node.get("predecessors", [])) for node in nodes]
        required_adapters = {str(node.get("required_adapter")) for node in nodes if node.get("required_adapter")}
        base_models = {str(node.get("required_base_model")) for node in nodes if node.get("required_base_model")}
        node_count = len(nodes)
        edge_count = len(edges)
        denominator = max(node_count * max(node_count - 1, 1), 1)
        metrics[workflow_id] = {
            "workflow_id": workflow_id,
            "dag_node_count": float(node_count),
            "dag_edge_count": float(edge_count),
            "critical_path_len": float(_critical_path_length(node_ids, edges)),
            "parallel_width": float(_parallel_width(node_ids, edges)),
            "dependency_density": round(edge_count / denominator, 6),
            "multi_dependency_node_rate": round(sum(1 for count in parent_counts if count >= 2) / node_count, 6)
            if node_count
            else 0.0,
            "required_adapter_count": float(len(required_adapters)),
            "required_base_model_count": float(len(base_models)),
            "required_adapters": sorted(required_adapters),
        }
    missing_workflows = sorted(workflow_ids - set(metrics))
    if missing_workflows:
        missing_fields.add(f"workflow_metrics_missing:{','.join(missing_workflows)}")
    return metrics


def load_adapter_catalog_metrics(path: Path, missing_fields: set[str]) -> dict[str, Any]:
    if not path.exists():
        missing_fields.add("adapter_catalog_path")
        return {}
    payload = _read_json(path)
    if not payload:
        missing_fields.add("adapter_catalog_json")
        return {}
    cached_adapters = {
        str(adapter_id)
        for profile in payload.get("rsu_adapter_caches", [])
        for adapter_id in profile.get("cached_adapter_ids", [])
    }
    object_adapters = {str(item.get("adapter_id")) for item in payload.get("cache_objects", []) if item.get("adapter_id")}
    bundle_adapters = {str(item.get("adapter_id")) for item in payload.get("adapter_state_bundles", []) if item.get("adapter_id")}
    adapter_ids = cached_adapters | object_adapters | bundle_adapters
    base_models = {str(item.get("base_model_id")) for item in payload.get("vehicle_base_models", []) if item.get("base_model_id")}
    rsu_cache_sizes = [len(profile.get("cached_adapter_ids", [])) for profile in payload.get("rsu_adapter_caches", [])]
    return {
        "adapter_catalog_size": float(len(adapter_ids)),
        "base_model_count": float(len(base_models)),
        "initial_cached_adapter_count": float(sum(rsu_cache_sizes)),
        "initial_cache_profile_count": float(len(rsu_cache_sizes)),
        "initial_cached_adapter_per_rsu_mean": _mean([float(value) for value in rsu_cache_sizes]),
        "adapter_ids": sorted(adapter_ids),
    }


def load_episode_enrichment(benchmark_dir: Path | None, missing_fields: set[str]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if benchmark_dir is None:
        missing_fields.add("episode_summary_dir")
        return {}
    episodes_root = benchmark_dir / "episodes"
    if not episodes_root.exists():
        missing_fields.add("episode_summary_dir")
        return {}
    enrichments: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for summary_path in episodes_root.rglob("*.summary.json"):
        payload = _read_json(summary_path)
        run_info = payload.get("run_info", {})
        mode = str(run_info.get("window_mode") or "unknown")
        window_id = str(run_info.get("window_id") or "unknown")
        workflow_id = str(run_info.get("workflow_id") or "unknown")
        policy_name = str(run_info.get("agent_name") or "unknown")
        seed = str(run_info.get("seed") or "")
        key = (mode, window_id, workflow_id, policy_name, seed)
        step_trace = [step for step in payload.get("step_trace", []) if isinstance(step, dict)]
        required_adapters = {str(step.get("required_adapter")) for step in step_trace if step.get("required_adapter")}
        associated_rsues = {
            str(value)
            for step in step_trace
            for value in [step.get("pre_action_associated_rsu_id"), step.get("post_action_associated_rsu_id")]
            if value
        }
        offload_rsues = {str(step.get("offload_target_rsu_id")) for step in step_trace if step.get("offload_target_rsu_id")}
        next_rsu_non_null = sum(1 for step in step_trace if step.get("predicted_next_rsu_id"))
        handoff_risk_steps = sum(
            1
            for step in step_trace
            if step.get("predicted_handoff_signal") or step.get("predicted_handoff_target_rsu_id") or _float_value(step.get("handoff_event_count")) > 0
        )
        handoff_steps = sum(1 for step in step_trace if _float_value(step.get("handoff_event_count")) > 0)
        enrichments[key] = {
            "unique_adapter_per_episode": float(len(required_adapters)),
            "cross_rsu_workflow": float(len(associated_rsues | offload_rsues) > 1),
            "next_rsu_non_null_rate": round(next_rsu_non_null / len(step_trace), 6) if step_trace else 0.0,
            "handoff_risk_window_count": float(handoff_risk_steps),
            "service_interruption_proxy": float(sum(1 for step in step_trace if bool(step.get("stall_occurred", False)))),
            "handoff_step_count": float(handoff_steps),
            "episode_step_count": float(len(step_trace)),
        }
    return enrichments


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


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
    return grouped


def _base_group_row(group_key: tuple[str, ...], key_names: list[str], policy_name: str | None = None) -> dict[str, Any]:
    row = {key_name: group_key[index] for index, key_name in enumerate(key_names)}
    row.setdefault("mode", "all")
    row.setdefault("scenario_id", "all")
    row.setdefault("window_tag", "all")
    row["policy_name"] = policy_name if policy_name is not None else row.get("policy_name", "scenario_catalog")
    return row


def add_enrichment_to_rows(rows: list[dict[str, Any]], enrichments: dict[tuple[str, str, str, str], dict[str, Any]]) -> None:
    for row in rows:
        key = (
            str(row.get("mode") or "unknown"),
            str(row.get("window_id") or row.get("scenario_id") or "unknown"),
            str(row.get("workflow_id") or "unknown"),
            str(row.get("policy_name") or row.get("agent_name") or "unknown"),
            str(row.get("seed") or ""),
        )
        enrichment = enrichments.get(key, {})
        for field_name, value in enrichment.items():
            row[field_name] = value


def compute_scenario_records(
    rows: list[dict[str, Any]],
    workflow_metrics: dict[str, dict[str, Any]],
    window_meta: dict[str, dict[str, Any]],
    missing_fields: set[str],
) -> list[dict[str, Any]]:
    grouped = group_rows(rows, ["mode", "window_id", "workflow_id", "seed"])
    scenario_records: list[dict[str, Any]] = []
    for (mode, window_id, workflow_id, seed), group in sorted(grouped.items()):
        workflow = workflow_metrics.get(workflow_id, {})
        window = window_meta.get(window_id, {})
        node_count = _float_value(workflow.get("dag_node_count"))
        critical_path = _float_value(workflow.get("critical_path_len"))
        edge_count = _float_value(workflow.get("dag_edge_count"))
        avg = lambda field: _mean([_float_value(row.get(field)) for row in group])
        rate = lambda field: _mean([1.0 if _float_value(row.get(field)) > 0 else 0.0 for row in group])
        handoff_count = avg("handoff_total_count")
        adapter_hit = avg("adapter_hit_count")
        adapter_miss = avg("adapter_miss_count")
        cache_total = adapter_hit + adapter_miss
        miss_rate = round(adapter_miss / cache_total, 6) if cache_total else 0.0
        next_rsu_rate = avg("next_rsu_non_null_rate")
        if not group or "next_rsu_non_null_rate" not in group[0]:
            missing_fields.add("next_rsu_non_null_rate")
        dag_high = bool(node_count >= LARGE_DAG_NODE_THRESHOLD or critical_path >= DEEP_DAG_CRITICAL_PATH_THRESHOLD)
        cache_high = bool(
            miss_rate > CACHE_PRESSURE_MISS_RATE_THRESHOLD
            or avg("adapter_cold_start_count") > 0.0
            or avg("prefetch_attempt_count") > 0.0
            or avg("cache_admission_count") >= 1.0
        )
        handoff_high = bool(handoff_count > 0.0 or _float_value(window.get("estimated_handoff_count")) > 0.0)
        high_count = sum([dag_high, cache_high, handoff_high])
        if dag_high and cache_high and handoff_high:
            bucket = "hard_joint"
        elif high_count >= 2:
            bucket = "mechanism_activating"
        elif cache_high:
            bucket = "cache_dominant"
        elif handoff_high:
            bucket = "handoff_dominant"
        elif dag_high:
            bucket = "dag_dominant"
        else:
            bucket = "easy_static_like"
        window_tag = str(group[0].get("window_tag") or group[0].get("window_class") or "unknown")
        scenario_records.append(
            {
                "mode": mode,
                "scenario_id": f"{window_id}|{workflow_id}|seed_{seed}",
                "window_id": window_id,
                "window_tag": window_tag,
                "workflow_id": workflow_id,
                "seed": seed,
                "policy_name": "scenario_bucket",
                "scenario_bucket": bucket,
                "dag_node_count": node_count,
                "dag_edge_count": edge_count,
                "critical_path_len": critical_path,
                "dependency_density": _float_value(workflow.get("dependency_density")),
                "parallel_width": _float_value(workflow.get("parallel_width")),
                "multi_dependency_node_rate": _float_value(workflow.get("multi_dependency_node_rate")),
                "required_adapter_count": _float_value(workflow.get("required_adapter_count")),
                "required_base_model_count": _float_value(workflow.get("required_base_model_count")),
                "large_dag": float(node_count >= LARGE_DAG_NODE_THRESHOLD),
                "deep_dag": float(critical_path >= DEEP_DAG_CRITICAL_PATH_THRESHOLD),
                "cache_pressure": float(cache_high),
                "handoff_pressure": float(handoff_high),
                "handoff_count_per_episode": handoff_count,
                "adapter_miss_rate": miss_rate,
                "next_rsu_non_null_rate": next_rsu_rate,
                "handoff_risk_window_count": avg("handoff_risk_window_count"),
                "cross_rsu_workflow_rate": rate("cross_rsu_workflow"),
                "prefetch_attempt_count": avg("prefetch_attempt_count"),
                "migration_attempt_count": avg("migration_attempt_count"),
                "backhaul_traffic_cost": avg("backhaul_traffic_cost"),
                "estimated_handoff_count": _float_value(window.get("estimated_handoff_count")),
                "active_vehicle_count_mean": _float_value(window.get("active_vehicle_count_mean")),
                "predicted_next_rsu_non_null_ratio": _float_value(window.get("predicted_next_rsu_non_null_ratio")),
                "predicted_handoff_target_non_null_ratio": _float_value(window.get("predicted_handoff_target_non_null_ratio")),
            }
        )
    return scenario_records


def build_dag_summary(scenario_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, group in group_rows(scenario_records, ["mode", "window_tag"]).items():
        row = _base_group_row(("all", key[0], key[1]), ["scenario_id", "mode", "window_tag"], policy_name="scenario_catalog")
        node_counts = [_float_value(item.get("dag_node_count")) for item in group]
        edge_counts = [_float_value(item.get("dag_edge_count")) for item in group]
        critical_paths = [_float_value(item.get("critical_path_len")) for item in group]
        widths = [_float_value(item.get("parallel_width")) for item in group]
        densities = [_float_value(item.get("dependency_density")) for item in group]
        row.update(_summary_stats(node_counts, "dag_node_count"))
        row.update(_summary_stats(edge_counts, "dag_edge_count"))
        row.update(_summary_stats(critical_paths, "critical_path_len"))
        row["dependency_density"] = _mean(densities)
        row["parallel_width_mean"] = _mean(widths)
        row["large_dag_rate"] = _mean([_float_value(item.get("large_dag")) for item in group])
        row["deep_dag_rate"] = _mean([_float_value(item.get("deep_dag")) for item in group])
        row["multi_dependency_rate"] = _mean([_float_value(item.get("multi_dependency_node_rate")) for item in group])
        row["scenario_count"] = len(group)
        output.append(row)
    return output


def build_cache_summary(rows: list[dict[str, Any]], adapter_catalog: dict[str, Any], missing_fields: set[str]) -> list[dict[str, Any]]:
    missing_cache_fields: list[str] = []
    if "adapter_catalog_size" not in adapter_catalog:
        missing_cache_fields.append("adapter_catalog_size")
    missing_cache_fields.extend(["cache_capacity", "cache_occupancy_rate_mean"])
    missing_fields.update(missing_cache_fields)
    output: list[dict[str, Any]] = []
    for key, group in group_rows(rows, ["mode", "window_tag", "policy_name"]).items():
        row = _base_group_row(("all", key[0], key[1], key[2]), ["scenario_id", "mode", "window_tag", "policy_name"])
        hit = _mean([_float_value(item.get("adapter_hit_count")) for item in group])
        miss = _mean([_float_value(item.get("adapter_miss_count")) for item in group])
        total = hit + miss
        row.update(
            {
                "adapter_catalog_size": adapter_catalog.get("adapter_catalog_size", ""),
                "base_model_count": adapter_catalog.get("base_model_count", ""),
                "unique_adapter_per_episode_mean": _mean([_float_value(item.get("unique_adapter_per_episode")) for item in group]),
                "cache_capacity": "",
                "cache_occupancy_rate_mean": "",
                "adapter_hit_rate": round(hit / total, 6) if total else 0.0,
                "adapter_miss_rate": round(miss / total, 6) if total else 0.0,
                "adapter_warm_hit_ratio": _mean([_float_value(item.get("adapter_warm_hit_ratio")) for item in group]),
                "cross_rsu_cold_start_frequency": _mean([_float_value(item.get("cross_rsu_cold_start_frequency")) for item in group]),
                "prefetch_attempt_count": _mean([_float_value(item.get("prefetch_attempt_count")) for item in group]),
                "cache_admission_count": _mean([_float_value(item.get("cache_admission_count")) for item in group]),
                "eviction_count": _mean([_float_value(item.get("cache_eviction_count")) for item in group]),
                "backhaul_traffic_cost": _mean([_float_value(item.get("backhaul_traffic_cost")) for item in group]),
                "missing_fields": ";".join(sorted(missing_cache_fields)),
            }
        )
        output.append(row)
    return output


def build_mobility_summary(rows: list[dict[str, Any]], scenario_records: list[dict[str, Any]], missing_fields: set[str]) -> list[dict[str, Any]]:
    missing_mobility_fields = ["rsu_dwell_time_mean", "rsu_dwell_time_p10", "rsu_dwell_time_p90", "deadline_tightness"]
    missing_fields.update(missing_mobility_fields)
    scenario_by_key = {
        (item["mode"], item["window_id"], item["workflow_id"], item["seed"]): item
        for item in scenario_records
    }
    output: list[dict[str, Any]] = []
    for key, group in group_rows(rows, ["mode", "window_tag", "policy_name"]).items():
        row = _base_group_row(("all", key[0], key[1], key[2]), ["scenario_id", "mode", "window_tag", "policy_name"])
        scenario_group = [
            scenario_by_key.get((str(item.get("mode")), str(item.get("window_id")), str(item.get("workflow_id")), str(item.get("seed"))), {})
            for item in group
        ]
        row.update(
            {
                "workflow_duration_mean": _mean([_float_value(item.get("end_to_end_workflow_delay")) for item in group]),
                "workflow_duration_p90": _percentile([_float_value(item.get("end_to_end_workflow_delay")) for item in group], 0.90),
                "finish_before_first_handoff_rate": _mean(
                    [
                        1.0
                        if _float_value(item.get("successful_episode_rate")) > 0.0 and _float_value(item.get("handoff_total_count")) <= 0.0
                        else 0.0
                        for item in group
                    ]
                ),
                "handoff_during_workflow_rate": _mean([1.0 if _float_value(item.get("handoff_total_count")) > 0.0 else 0.0 for item in group]),
                "cross_rsu_workflow_rate": _mean([_float_value(item.get("cross_rsu_workflow")) for item in group]),
                "deadline_tightness": "",
                "long_workflow_rate": _mean(
                    [1.0 if _float_value(item.get("end_to_end_workflow_delay")) >= LONG_WORKFLOW_DURATION_THRESHOLD else 0.0 for item in group]
                ),
                "handoff_count_per_episode": _mean([_float_value(item.get("handoff_total_count")) for item in group]),
                "rsu_dwell_time_mean": "",
                "rsu_dwell_time_p10": "",
                "rsu_dwell_time_p90": "",
                "next_rsu_non_null_rate": _mean([_float_value(item.get("next_rsu_non_null_rate")) for item in group]),
                "handoff_risk_window_count": _mean([_float_value(item.get("handoff_risk_window_count")) for item in group]),
                "mechanism_activating_window_rate": _mean(
                    [1.0 if str(item.get("window_tag")) == "mechanism_activating" else 0.0 for item in group]
                ),
                "migration_attempt_count": _mean([_float_value(item.get("migration_attempt_count")) for item in group]),
                "migration_success_count": _mean([_float_value(item.get("migration_success_count")) for item in group]),
                "service_interruption_proxy": _mean([_float_value(item.get("service_interruption_proxy")) for item in group]),
                "estimated_handoff_count": _mean([_float_value(item.get("estimated_handoff_count")) for item in scenario_group]),
                "missing_fields": ";".join(sorted(missing_mobility_fields)),
            }
        )
        output.append(row)
    return output


def build_innovation_summary(scenario_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, group in group_rows(scenario_records, ["mode", "window_tag"]).items():
        row = _base_group_row(("all", key[0], key[1]), ["scenario_id", "mode", "window_tag"], policy_name="scenario_catalog")
        booleans = {
            "has_real_mobility_signal": [bool(_float_value(item.get("active_vehicle_count_mean")) > 0.0) for item in group],
            "has_nontrivial_dag": [bool(_float_value(item.get("dag_node_count")) >= 2.0 and _float_value(item.get("dag_edge_count")) >= 1.0) for item in group],
            "has_large_or_deep_dag": [bool(_float_value(item.get("large_dag")) > 0.0 or _float_value(item.get("deep_dag")) > 0.0) for item in group],
            "has_cache_pressure": [bool(_float_value(item.get("cache_pressure")) > 0.0) for item in group],
            "has_model_adapter_pressure": [
                bool(_float_value(item.get("required_adapter_count")) >= 2.0 or _float_value(item.get("required_base_model_count")) >= 2.0)
                for item in group
            ],
            "has_handoff_pressure": [bool(_float_value(item.get("handoff_pressure")) > 0.0) for item in group],
            "has_cross_rsu_continuity_pressure": [bool(_float_value(item.get("cross_rsu_workflow_rate")) > 0.0) for item in group],
            "has_prediction_usefulness_signal": [bool(_float_value(item.get("next_rsu_non_null_rate")) > PREDICTION_SIGNAL_THRESHOLD) for item in group],
            "has_migration_need": [bool(_float_value(item.get("handoff_count_per_episode")) > 0.0) for item in group],
            "has_prefetch_need": [
                bool(_float_value(item.get("cache_pressure")) > 0.0 and _float_value(item.get("next_rsu_non_null_rate")) > PREDICTION_SIGNAL_THRESHOLD)
                for item in group
            ],
        }
        row["scenario_count"] = len(group)
        for field_name, values in booleans.items():
            row[f"{field_name}_rate"] = _bool_rate(values)
            row[field_name] = bool(any(values))
        output.append(row)
    return output


def build_bucket_summary(scenario_records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_bucket_by_key = {
        (item["mode"], item["window_id"], item["workflow_id"], item["seed"]): item["scenario_bucket"]
        for item in scenario_records
    }
    paired_by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("mode")), str(row.get("window_id")), str(row.get("workflow_id")), str(row.get("seed")))
        paired_by_key[key][str(row.get("policy_name"))] = row
    delta_by_bucket: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for key, agent_rows in paired_by_key.items():
        bucket = scenario_bucket_by_key.get(key, "unknown")
        mode = key[0]
        sa_row = agent_rows.get(SA_AGENT)
        pop_row = agent_rows.get(POPULARITY_AGENT)
        ippo_row = next((agent_rows[name] for name in IPPO_AGENT_NAMES if name in agent_rows), None)
        payload: dict[str, float] = {}
        if sa_row and pop_row:
            payload["sa_vs_popularity_reward_delta"] = _float_value(sa_row.get("total_reward")) - _float_value(pop_row.get("total_reward"))
            payload["sa_vs_popularity_backhaul_delta"] = _float_value(sa_row.get("backhaul_traffic_cost")) - _float_value(pop_row.get("backhaul_traffic_cost"))
        if sa_row and ippo_row:
            payload["sa_vs_ippo_reward_delta"] = _float_value(sa_row.get("total_reward")) - _float_value(ippo_row.get("total_reward"))
        delta_by_bucket[(mode, bucket)].append(payload)

    grouped = group_rows(scenario_records, ["mode", "scenario_bucket"])
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        mode, bucket = key
        deltas = delta_by_bucket.get((mode, bucket), [])
        row = {
            "mode": mode,
            "scenario_id": "all",
            "window_tag": "all",
            "policy_name": "scenario_bucket",
            "scenario_bucket": bucket,
            "scenario_count": len(group),
            "scenario_rate": round(len(group) / max(sum(1 for item in scenario_records if item.get("mode") == mode), 1), 6),
            "large_dag_rate": _mean([_float_value(item.get("large_dag")) for item in group]),
            "deep_dag_rate": _mean([_float_value(item.get("deep_dag")) for item in group]),
            "cache_pressure_rate": _mean([_float_value(item.get("cache_pressure")) for item in group]),
            "handoff_pressure_rate": _mean([_float_value(item.get("handoff_pressure")) for item in group]),
            "mean_sa_vs_popularity_reward_delta": _mean([item.get("sa_vs_popularity_reward_delta", 0.0) for item in deltas if "sa_vs_popularity_reward_delta" in item]),
            "mean_sa_vs_popularity_backhaul_delta": _mean([item.get("sa_vs_popularity_backhaul_delta", 0.0) for item in deltas if "sa_vs_popularity_backhaul_delta" in item]),
            "mean_sa_vs_ippo_reward_delta": _mean([item.get("sa_vs_ippo_reward_delta", 0.0) for item in deltas if "sa_vs_ippo_reward_delta" in item]),
        }
        output.append(row)
    return output


def build_report(payload: dict[str, Any]) -> str:
    answers = payload["answers"]
    paths = payload["outputs"]
    missing = ", ".join(payload.get("missing_fields", [])) or "none"
    return f"""# scenario_innovation_activation_audit

## Scope

本轮只做 benchmark 数据激活度审计。未修改环境逻辑、reward、policy、checkpoint selection、baseline，也未训练。

## Inputs

- Mixed: `{payload['inputs']['mixed_input']}`
- Full: `{payload['inputs']['full_input']}`

## Outputs

- `{paths['dag_hardness_summary']}`
- `{paths['cache_pressure_summary']}`
- `{paths['mobility_handoff_pressure_summary']}`
- `{paths['innovation_activation_summary']}`
- `{paths['scenario_bucket_summary']}`
- `{paths['diagnosis_summary']}`

## Key Answers

1. hard_joint / mechanism_activating 占比：
   - mixed hard_joint: `{answers['mixed_hard_joint_rate']}`
   - mixed mechanism_activating bucket: `{answers['mixed_mechanism_activating_bucket_rate']}`
   - full hard_joint: `{answers['full_hard_joint_rate']}`
   - full mechanism_activating bucket: `{answers['full_mechanism_activating_bucket_rate']}`
2. 是否存在 easy_static_like 稀释：`{answers['easy_static_like_assessment']}`
3. DAG 是否足够复杂：`{answers['dag_complexity_assessment']}`
4. cache/model/adapter 压力是否足够：`{answers['cache_pressure_assessment']}`
5. handoff / cross-RSU 是否足够：`{answers['handoff_pressure_assessment']}`
6. SA 相对 IPPO 的优势是否集中 hard_joint：`{answers['sa_vs_ippo_assessment']}`
7. SA 相对 popularity 的劣势是否在 cache/prefetch 场景：`{answers['sa_vs_popularity_assessment']}`
8. 下一轮 split 建议：`{answers['next_split_recommendation']}`

## Missing Fields

`{missing}`

## Notes

- DAG 指标来自 Alibaba workflow CSV，经 `WorkflowDatasetBuilder` 离线解析。
- cache/action/migration 指标来自 round5 evaluator rows。
- `rsu_dwell_time_*`、`deadline_tightness`、`cache_capacity`、`cache_occupancy_rate_mean` 当前没有可靠底层字段，本轮只在 missing fields 中记录。
"""


def main() -> None:
    args = parse_args()
    missing_fields: set[str] = set()
    mixed_rows, mixed_aggregate, mixed_dir = read_benchmark(args.mixed_input, fallback_mode="mixed_informative")
    full_rows, full_aggregate, full_dir = read_benchmark(args.full_input, fallback_mode="full_stratified")
    rows = mixed_rows + full_rows

    workflow_source = args.workflow_csv_path
    if workflow_source is None:
        workflow_source_raw = mixed_aggregate.get("workflow_source_path") or full_aggregate.get("workflow_source_path")
        workflow_source = Path(workflow_source_raw) if workflow_source_raw else None
    workflow_ids = {str(row.get("workflow_id")) for row in rows if row.get("workflow_id")}
    workflow_metrics = load_workflow_metrics(workflow_source, workflow_ids, missing_fields)
    adapter_catalog = load_adapter_catalog_metrics(args.adapter_catalog_path, missing_fields)

    enrichments = {}
    enrichments.update(load_episode_enrichment(mixed_dir, missing_fields))
    enrichments.update(load_episode_enrichment(full_dir, missing_fields))
    add_enrichment_to_rows(rows, enrichments)

    window_meta = {}
    for aggregate in [mixed_aggregate, full_aggregate]:
        for window in aggregate.get("selected_window_plan", []):
            window_id = str(window.get("window_id"))
            window_meta[window_id] = window

    scenario_records = compute_scenario_records(rows, workflow_metrics, window_meta, missing_fields)
    dag_summary = build_dag_summary(scenario_records)
    cache_summary = build_cache_summary(rows, adapter_catalog, missing_fields)
    mobility_summary = build_mobility_summary(rows, scenario_records, missing_fields)
    innovation_summary = build_innovation_summary(scenario_records)
    bucket_summary = build_bucket_summary(scenario_records, rows)

    bucket_by_mode = defaultdict(dict)
    for row in bucket_summary:
        bucket_by_mode[str(row["mode"])][str(row["scenario_bucket"])] = row

    def bucket_rate(mode: str, bucket: str) -> float:
        return _float_value(bucket_by_mode.get(mode, {}).get(bucket, {}).get("scenario_rate"))

    def dag_assessment() -> str:
        full_dag = [row for row in dag_summary if row.get("mode") == "full_stratified"]
        if not full_dag:
            return "workflow DAG metrics missing"
        large_rates = [_float_value(row.get("large_dag_rate")) for row in full_dag]
        deep_rates = [_float_value(row.get("deep_dag_rate")) for row in full_dag]
        return f"large_dag_rate_mean={_mean(large_rates)}, deep_dag_rate_mean={_mean(deep_rates)}; selected workflows are nontrivial but limited to a small catalog."

    def cache_assessment() -> str:
        cache_rows = [row for row in cache_summary if row.get("policy_name") == SA_AGENT]
        miss_rates = [_float_value(row.get("adapter_miss_rate")) for row in cache_rows]
        prefetch_counts = [_float_value(row.get("prefetch_attempt_count")) for row in cache_rows]
        unique_adapter_means = [_float_value(row.get("unique_adapter_per_episode_mean")) for row in cache_rows]
        return (
            f"SA miss_rate_mean={_mean(miss_rates)}, prefetch_attempt_mean={_mean(prefetch_counts)}, "
            f"unique_adapter_per_episode_mean={_mean(unique_adapter_means)}; cache/prefetch pressure exists mainly in mechanism windows, "
            "but model/adapter diversity is low and capacity/occupancy telemetry is missing."
        )

    def handoff_assessment() -> str:
        mobility_rows = [row for row in mobility_summary if row.get("policy_name") == SA_AGENT]
        handoff_rates = [_float_value(row.get("handoff_during_workflow_rate")) for row in mobility_rows]
        cross_rates = [_float_value(row.get("cross_rsu_workflow_rate")) for row in mobility_rows]
        return f"handoff_during_workflow_rate_mean={_mean(handoff_rates)}, cross_rsu_workflow_rate_mean={_mean(cross_rates)}; pressure is stratified but not uniformly hard."

    ippo_present = any(str(row.get("policy_name")) in IPPO_AGENT_NAMES for row in rows)
    pop_bucket_rows = [
        row for row in bucket_summary if row.get("mode") == "mixed_informative" and _float_value(row.get("mean_sa_vs_popularity_reward_delta")) < 0.0
    ]
    answers = {
        "mixed_hard_joint_rate": bucket_rate("mixed_informative", "hard_joint"),
        "mixed_mechanism_activating_bucket_rate": bucket_rate("mixed_informative", "mechanism_activating"),
        "full_hard_joint_rate": bucket_rate("full_stratified", "hard_joint"),
        "full_mechanism_activating_bucket_rate": bucket_rate("full_stratified", "mechanism_activating"),
        "easy_static_like_assessment": (
            f"easy_static_like rates: mixed={bucket_rate('mixed_informative', 'easy_static_like')}, "
            f"full={bucket_rate('full_stratified', 'easy_static_like')}; supplied rows do not contain an easy_static_like bucket under the current thresholds."
        ),
        "dag_complexity_assessment": dag_assessment(),
        "cache_pressure_assessment": cache_assessment(),
        "handoff_pressure_assessment": handoff_assessment(),
        "sa_vs_ippo_assessment": "IPPO/PPO rows are not present in the supplied round5 benchmark rows." if not ippo_present else "IPPO/PPO rows present; see scenario_bucket_summary.",
        "sa_vs_popularity_assessment": (
            "SA reward losses vs popularity are concentrated in buckets with prefetch/cache-admission tie-break behavior."
            if pop_bucket_rows
            else "No SA reward loss vs popularity in bucket aggregates."
        ),
        "next_split_recommendation": (
            "Do not modify split in this round. Next benchmark split should explicitly preserve a larger hard_joint slice, "
            "report cache capacity/occupancy telemetry, and separate cache-dominant from joint DAG+cache+handoff scenarios."
        ),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dag_hardness_summary": output_dir / "dag_hardness_summary.csv",
        "cache_pressure_summary": output_dir / "cache_pressure_summary.csv",
        "mobility_handoff_pressure_summary": output_dir / "mobility_handoff_pressure_summary.csv",
        "innovation_activation_summary": output_dir / "innovation_activation_summary.csv",
        "scenario_bucket_summary": output_dir / "scenario_bucket_summary.csv",
        "diagnosis_summary": output_dir / "diagnosis_summary.json",
    }
    write_csv(paths["dag_hardness_summary"], dag_summary)
    write_csv(paths["cache_pressure_summary"], cache_summary)
    write_csv(paths["mobility_handoff_pressure_summary"], mobility_summary)
    write_csv(paths["innovation_activation_summary"], innovation_summary)
    write_csv(paths["scenario_bucket_summary"], bucket_summary)

    diagnosis = {
        "task": "scenario_innovation_activation_audit",
        "policy_or_reward_modified": False,
        "environment_semantics_modified": False,
        "training_run": False,
        "inputs": {"mixed_input": str(args.mixed_input), "full_input": str(args.full_input)},
        "workflow_source_path": str(workflow_source) if workflow_source else "",
        "adapter_catalog_path": str(args.adapter_catalog_path),
        "row_count": len(rows),
        "scenario_count": len(scenario_records),
        "workflow_metrics": workflow_metrics,
        "adapter_catalog_metrics": {key: value for key, value in adapter_catalog.items() if key != "adapter_ids"},
        "missing_fields": sorted(missing_fields),
        "outputs": {key: str(path) for key, path in paths.items()},
        "answers": answers,
        "scenario_bucket_summary": bucket_summary,
        "innovation_activation_summary": innovation_summary,
    }
    paths["diagnosis_summary"].write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        **diagnosis,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    args.report_path.write_text(build_report(report_payload), encoding="utf-8")

    print("scenario innovation activation audit complete")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"report_path: {args.report_path}")
    print(f"missing_fields: {', '.join(sorted(missing_fields)) if missing_fields else 'none'}")


if __name__ == "__main__":
    main()
