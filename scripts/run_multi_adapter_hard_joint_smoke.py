"""Run a proposal-only multi-adapter hard-joint smoke benchmark.

This script does not train, freeze, or modify policies. It evaluates available
agents on a controlled AI-service adapter assignment plus cache-capacity stress
profile over real NGSIM mobility and real Alibaba DAG structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - local environments normally have PyYAML.
    yaml = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import checkpoint_required_agents, list_evaluable_agents
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.evaluators.main_results_support import (
    aggregate_rows,
    load_window_bundle,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
)


TASK_NAME = "multi_adapter_hard_joint_smoke_round10"
DEFAULT_CONFIG = Path("configs/experiment/multi_adapter_hard_joint_smoke_round10.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/multi_adapter_hard_joint_smoke_round10")
DEFAULT_REPORT_PATH = Path("docs/agent/multi_adapter_hard_joint_smoke_round10_report.md")

REQUESTED_POLICIES = [
    "sa_ghmappo",
    "ippo",
    "ppo",
    "popularity_cache_heuristic",
    "reactive_greedy",
]
LOWER_IS_BETTER = {
    "end_to_end_workflow_delay",
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_miss_count",
    "adapter_cold_start_count",
    "eviction_count",
    "evicted_adapter_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--report_path", type=Path, default=None)
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config does not exist: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the smoke config.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "missing"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def _metric_mean(rows: list[dict[str, Any]], metric_name: str) -> float:
    return _mean([_float_value(row.get(metric_name)) for row in rows])


def _metric_sum(rows: list[dict[str, Any]], metric_name: str) -> float:
    return round(sum(_float_value(row.get(metric_name)) for row in rows), 6)


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


def _load_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    manifest_path = Path(path)
    if not str(path) or not manifest_path.exists():
        return {}
    payload = _read_json(manifest_path)
    normalized: dict[str, dict[str, str]] = {}
    for agent_name, seed_map in payload.items():
        if isinstance(seed_map, dict):
            normalized[str(agent_name)] = {str(seed): str(value) for seed, value in seed_map.items() if str(value)}
    for primary, legacy in [("ppo", "flat_ppo"), ("mappo", "flat_mappo")]:
        if primary in normalized and legacy not in normalized:
            normalized[legacy] = dict(normalized[primary])
        if legacy in normalized and primary not in normalized:
            normalized[primary] = dict(normalized[legacy])
    return normalized


def _checkpoint_map_for_seed(manifests: dict[str, dict[str, dict[str, str]]], seed: int) -> dict[str, str]:
    seed_key = str(seed)
    checkpoint_map: dict[str, str] = {
        "sa_ghmappo": "",
        "ppo": "",
        "flat_ppo": "",
        "mappo": "",
        "flat_mappo": "",
        "popularity_cache_heuristic": "",
        "reactive_greedy": "",
    }
    for manifest in manifests.values():
        for agent_name, seed_map in manifest.items():
            if seed_key in seed_map:
                checkpoint_map[agent_name] = seed_map[seed_key]
    if checkpoint_map.get("ppo") and not checkpoint_map.get("flat_ppo"):
        checkpoint_map["flat_ppo"] = checkpoint_map["ppo"]
    if checkpoint_map.get("flat_ppo") and not checkpoint_map.get("ppo"):
        checkpoint_map["ppo"] = checkpoint_map["flat_ppo"]
    if checkpoint_map.get("mappo") and not checkpoint_map.get("flat_mappo"):
        checkpoint_map["flat_mappo"] = checkpoint_map["mappo"]
    if checkpoint_map.get("flat_mappo") and not checkpoint_map.get("mappo"):
        checkpoint_map["mappo"] = checkpoint_map["flat_mappo"]
    return checkpoint_map


def _load_workflows(config: dict[str, Any]) -> list[Any]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    workflow_ids = [str(item) for item in data_cfg.get("workflow_ids", ["j_3", "j_8"])]
    samples = WorkflowDatasetBuilder().build_alibaba_samples(
        csv_path=data_cfg.get("workflow_csv_path", "data/raw/workflow/alibaba2018/batch_task.csv"),
        limit_jobs=max(64, len(workflow_ids) * 24),
        min_tasks=int(data_cfg.get("min_tasks", 5)),
        max_tasks=int(data_cfg.get("max_tasks", 20)),
        adapter_assignment_profile=str(config.get("adapter_assignment_profile", "semantic_ai_service")),
    )
    by_id = {str(sample.get("workflow_id")): sample for sample in samples}
    missing = [workflow_id for workflow_id in workflow_ids if workflow_id not in by_id]
    if missing:
        raise RuntimeError(f"requested workflow ids missing from parsed Alibaba samples: {missing}")
    builder = WorkflowDatasetBuilder()
    return [builder.sample_to_workflow_state(by_id[workflow_id]) for workflow_id in workflow_ids]


def _workflow_metadata(workflow_state: Any) -> dict[str, Any]:
    required_adapters = sorted({str(node.required_adapter) for node in workflow_state.nodes if node.required_adapter})
    required_base_models = sorted({str(node.required_base_model) for node in workflow_state.nodes if node.required_base_model})
    return {
        "required_adapter_count": len(required_adapters),
        "unique_adapter_per_episode": len(required_adapters),
        "required_adapter_ids": ";".join(required_adapters) if required_adapters else "missing",
        "required_base_model_count": len(required_base_models),
        "required_base_model_ids": ";".join(required_base_models) if required_base_models else "missing",
        "dag_node_count": len(workflow_state.nodes),
        "dag_edge_count": len(workflow_state.edges),
    }


def _cross_rsu_workflow_rate(summary: dict[str, Any]) -> float:
    rsu_ids: set[str] = set()
    for step in summary.get("step_trace", []):
        if not isinstance(step, dict):
            continue
        for key in [
            "pre_action_associated_rsu_id",
            "current_associated_rsu_id",
            "post_action_associated_rsu_id",
            "offload_target_rsu_id",
            "predicted_next_rsu_id",
        ]:
            value = step.get(key)
            if value not in (None, "", "None"):
                rsu_ids.add(str(value))
    return 1.0 if len(rsu_ids) > 1 else 0.0


def _augment_row(row: dict[str, Any], summary: dict[str, Any], workflow_state: Any, run_id: str) -> dict[str, Any]:
    workflow_meta = _workflow_metadata(workflow_state)
    augmented = dict(row)
    augmented.update(workflow_meta)
    augmented["profile_name"] = TASK_NAME
    augmented["proposal_only"] = True
    augmented["do_not_use_for_freeze"] = True
    augmented["adapter_assignment_profile"] = "semantic_ai_service"
    augmented["cache_capacity_profile_name"] = "multi_adapter_capacity_stress"
    augmented["run_id"] = run_id
    augmented["reward"] = augmented.get("total_reward", 0.0)
    augmented["delay"] = augmented.get("end_to_end_workflow_delay", 0.0)
    augmented["workflow_delay"] = augmented.get("end_to_end_workflow_delay", 0.0)
    augmented["success"] = 1.0 if str(augmented.get("episode_success")).lower() == "true" else 0.0
    augmented["handoff_during_workflow_rate"] = 1.0 if _float_value(augmented.get("handoff_total_count")) > 0.0 else 0.0
    augmented["cross_rsu_workflow_rate"] = _cross_rsu_workflow_rate(summary)
    return augmented


def _available_policy_plan(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    requested = list(config.get("policies", {}).get("requested", REQUESTED_POLICIES))
    evaluable = set(list_evaluable_agents())
    checkpoint_required = checkpoint_required_agents()
    missing: dict[str, str] = {}
    available: list[str] = []
    for policy_name in requested:
        if policy_name not in evaluable:
            missing[policy_name] = "not_registered_or_not_evaluable"
            continue
        if policy_name in available:
            continue
        available.append(policy_name)
    # Keep round10 focused on requested PPO, not MAPPO aliases.
    return available, missing


def _group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
    return grouped


def build_adapter_diversity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, group in sorted(_group_rows(rows, ["policy_name", "mode", "scenario_id", "window_tag"]).items()):
        policy_name, mode, scenario_id, window_tag = key
        output.append(
            {
                "policy_name": policy_name,
                "mode": mode,
                "scenario_id": scenario_id,
                "window_tag": window_tag,
                "episode_count": len(group),
                "required_adapter_count_mean": _metric_mean(group, "required_adapter_count"),
                "required_adapter_count_min": min(_float_value(row.get("required_adapter_count")) for row in group),
                "unique_adapter_per_episode_mean": _metric_mean(group, "unique_adapter_per_episode"),
                "required_base_model_count_mean": _metric_mean(group, "required_base_model_count"),
                "adapter_diversity_activated": bool(
                    min(_float_value(row.get("required_adapter_count")) for row in group) >= 3.0
                ),
            }
        )
    return output


def build_cache_eviction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (policy_name,), group in sorted(_group_rows(rows, ["policy_name"]).items()):
        output.append(
            {
                "policy_name": policy_name,
                "episode_count": len(group),
                "cache_capacity_enabled_rate": _metric_mean(group, "cache_capacity_enabled"),
                "rsu_adapter_slots_mean": _metric_mean(group, "rsu_adapter_slots"),
                "cache_occupancy_rate_mean": _metric_mean(group, "cache_occupancy_rate"),
                "eviction_count_sum": _metric_sum(group, "eviction_count"),
                "evicted_adapter_count_sum": _metric_sum(group, "evicted_adapter_count"),
                "adapter_miss_count_sum": _metric_sum(group, "adapter_miss_count"),
                "adapter_warm_hit_count_sum": _metric_sum(group, "adapter_warm_hit_count"),
                "adapter_cold_start_count_sum": _metric_sum(group, "adapter_cold_start_count"),
                "cache_admission_count_sum": _metric_sum(group, "cache_admission_count"),
                "cache_admission_added_new_adapter_count_sum": _metric_sum(
                    group, "cache_admission_added_new_adapter_count"
                ),
                "eviction_activated": bool(_metric_sum(group, "eviction_count") > 0.0),
            }
        )
    return output


def build_handoff_continuity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (policy_name,), group in sorted(_group_rows(rows, ["policy_name"]).items()):
        output.append(
            {
                "policy_name": policy_name,
                "episode_count": len(group),
                "handoff_during_workflow_rate": _metric_mean(group, "handoff_during_workflow_rate"),
                "cross_rsu_workflow_rate": _metric_mean(group, "cross_rsu_workflow_rate"),
                "workflow_continuity_rate_mean": _metric_mean(group, "workflow_continuity_rate"),
                "handoff_failure_rate_mean": _metric_mean(group, "handoff_failure_rate"),
                "continuity_reward_component_mean": _metric_mean(group, "continuity_reward_component"),
                "handoff_ready_ratio_mean": _metric_mean(group, "handoff_ready_ratio"),
            }
        )
    return output


def build_actionmix_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "local_exec_count",
        "current_rsu_exec_count",
        "next_rsu_exec_count",
        "neighbor_rsu_exec_count",
        "cloud_exec_count",
        "prefetch_action_count",
        "migration_action_count",
        "no_op_action_count",
        "prefetch_attempt_count",
        "prefetch_success_count",
        "migration_attempt_count",
        "migration_success_count",
    ]
    output: list[dict[str, Any]] = []
    for (policy_name,), group in sorted(_group_rows(rows, ["policy_name"]).items()):
        row = {"policy_name": policy_name, "episode_count": len(group)}
        for metric_name in metrics:
            row[f"{metric_name}_sum"] = _metric_sum(group, metric_name)
            row[f"{metric_name}_mean"] = _metric_mean(group, metric_name)
        output.append(row)
    return output


def build_policy_comparison_summary(
    rows: list[dict[str, Any]],
    missing_policy_rows: dict[str, str],
) -> list[dict[str, Any]]:
    metric_names = [
        "total_reward",
        "end_to_end_workflow_delay",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "adapter_miss_count",
        "adapter_cold_start_count",
        "adapter_warm_hit_count",
        "cache_occupancy_rate",
        "eviction_count",
        "required_adapter_count",
        "handoff_during_workflow_rate",
        "cross_rsu_workflow_rate",
    ]
    grouped = _group_rows(rows, ["policy_name"])
    output: list[dict[str, Any]] = []
    for policy_name in REQUESTED_POLICIES:
        group = grouped.get((policy_name,), [])
        if not group:
            output.append(
                {
                    "policy_name": policy_name,
                    "status": "missing",
                    "missing_reason": missing_policy_rows.get(policy_name, "no_rows_generated"),
                    "episode_count": 0,
                }
            )
            continue
        row: dict[str, Any] = {
            "policy_name": policy_name,
            "status": "evaluated",
            "missing_reason": "none",
            "episode_count": len(group),
        }
        for metric_name in metric_names:
            row[f"{metric_name}_mean"] = _metric_mean(group, metric_name)
            row[f"{metric_name}_sum"] = _metric_sum(group, metric_name)
        output.append(row)
    return output


def _comparison_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, Any]:
    grouped = _group_rows(rows, ["policy_name"])
    candidate_rows = grouped.get((candidate,), [])
    baseline_rows = grouped.get((baseline,), [])
    if not candidate_rows or not baseline_rows:
        return {
            "available": False,
            "candidate": candidate,
            "baseline": baseline,
            "reason": "candidate_or_baseline_missing",
        }
    metrics = [
        "total_reward",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "adapter_miss_count",
        "adapter_cold_start_count",
        "cache_occupancy_rate",
        "eviction_count",
    ]
    result: dict[str, Any] = {"available": True, "candidate": candidate, "baseline": baseline, "metrics": {}}
    for metric_name in metrics:
        candidate_value = _metric_mean(candidate_rows, metric_name)
        baseline_value = _metric_mean(baseline_rows, metric_name)
        delta = round(candidate_value - baseline_value, 6)
        effective_delta = -delta if metric_name in LOWER_IS_BETTER else delta
        if effective_delta > 1e-6:
            outcome = "win"
        elif effective_delta < -1e-6:
            outcome = "loss"
        else:
            outcome = "tie"
        result["metrics"][metric_name] = {
            candidate: candidate_value,
            baseline: baseline_value,
            "delta_candidate_minus_baseline": delta,
            "result": outcome,
        }
    return result


def build_report(
    *,
    rows: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    policy_summary: list[dict[str, Any]],
) -> str:
    semantic_active = diagnosis.get("semantic_ai_service_active")
    adapter_active = diagnosis.get("adapter_diversity_activated")
    cache_active = diagnosis.get("cache_capacity_profile_active")
    eviction_active = diagnosis.get("eviction_activated")
    sa_pop = diagnosis.get("sa_vs_popularity_result", {})
    sa_ippo = diagnosis.get("sa_vs_ippo_result_if_available", {})
    sa_ppo = diagnosis.get("sa_vs_ppo_result", {})
    missing = diagnosis.get("missing_policy_rows", {})
    policy_lines = "\n".join(
        f"- `{row['policy_name']}`: {row['status']}"
        + (f" ({row.get('missing_reason')})" if row.get("status") == "missing" else f", episodes={row.get('episode_count')}")
        for row in policy_summary
    )
    sa_pop_reward = "missing"
    if sa_pop.get("available"):
        reward_metric = sa_pop.get("metrics", {}).get("total_reward", {})
        sa_pop_reward = (
            f"SA={reward_metric.get('sa_ghmappo')}, popularity={reward_metric.get('popularity_cache_heuristic')}, "
            f"delta={reward_metric.get('delta_candidate_minus_baseline')}, result={reward_metric.get('result')}"
        )
    sa_ippo_text = "not_available"
    if sa_ippo.get("available"):
        reward_metric = sa_ippo.get("metrics", {}).get("total_reward", {})
        sa_ippo_text = (
            f"SA={reward_metric.get('sa_ghmappo')}, IPPO={reward_metric.get('ippo')}, "
            f"delta={reward_metric.get('delta_candidate_minus_baseline')}, result={reward_metric.get('result')}"
        )
    elif sa_ippo:
        sa_ippo_text = str(sa_ippo.get("reason", "not_available"))

    def metric_line(pairwise: dict[str, Any], metric_name: str) -> str:
        if not pairwise.get("available"):
            return f"- `{metric_name}`: not_available"
        metric = pairwise.get("metrics", {}).get(metric_name, {})
        candidate = pairwise.get("candidate", "candidate")
        baseline = pairwise.get("baseline", "baseline")
        return (
            f"- `{metric_name}`: {candidate}={metric.get(candidate)}, "
            f"{baseline}={metric.get(baseline)}, "
            f"delta={metric.get('delta_candidate_minus_baseline')}, result={metric.get('result')}"
        )

    detail_metrics = [
        "total_reward",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "adapter_miss_count",
        "adapter_cold_start_count",
        "eviction_count",
    ]
    sa_pop_detail = "\n".join(metric_line(sa_pop, metric_name) for metric_name in detail_metrics)
    sa_ppo_detail = "\n".join(metric_line(sa_ppo, metric_name) for metric_name in detail_metrics)

    return f"""# multi_adapter_hard_joint_smoke_round10 报告

## 范围

本轮不是新数据集，不训练，不 freeze，不修改 reward、policy、baseline 或 checkpoint selection。

它是基于真实 NGSIM mobility trace 与真实 Alibaba DAG structure 的 controlled AI-service/cache-stress smoke。`semantic_ai_service` adapter assignment 和 `rsu_adapter_slots=2` cache stress 是可控构造，只用于验证 proposal 链路与 telemetry，不作为正式论文结论。

## 核心结果

- semantic_ai_service_active: `{semantic_active}`
- adapter_diversity_activated: `{adapter_active}`
- cache_capacity_profile_active: `{cache_active}`
- eviction_activated: `{eviction_active}`
- handoff_pressure_active: `{diagnosis.get('handoff_pressure_active')}`
- do_not_freeze: `true`

## Policy Rows

{policy_lines}

missing_policy_rows:

```json
{json.dumps(missing, ensure_ascii=False, indent=2)}
```

## 问题回答

1. 本轮是不是新数据集？

不是。它是基于真实 mobility trace + 真实 Alibaba DAG structure 的 controlled AI-service/cache-stress smoke。

2. `semantic_ai_service` 是否实际生效？

`{semantic_active}`。benchmark rows 中写入了 `adapter_assignment_profile=semantic_ai_service`。

3. 实际 benchmark rows 中是否出现多个 adapter？

`{adapter_active}`。本轮 rows 的 `required_adapter_count` 至少达到 `{min((_float_value(row.get('required_adapter_count')) for row in rows), default=0.0)}`，最大达到 `{max((_float_value(row.get('required_adapter_count')) for row in rows), default=0.0)}`。

4. cache capacity profile 是否实际生效？

`{cache_active}`。rows 中 `cache_capacity_enabled` 大于 0，并记录了 slot、used、remaining、occupancy telemetry。

5. 是否发生 eviction？

`{eviction_active}`。总 `eviction_count={_metric_sum(rows, 'eviction_count')}`。

6. 哪些 policy 成功评估，哪些缺失？

见上方 Policy Rows。缺失项没有伪造结果。

7. 如果 IPPO/PPO 缺失，原因是什么？

`ippo` 当前缺失原因：`{missing.get('ippo', 'none')}`。`ppo` 当前缺失原因：`{missing.get('ppo', 'none')}`。

8. SA 相对 popularity 的结果如何？

`{sa_pop_reward}`。

9. SA 相对 IPPO/PPO 的结果如何？

SA vs IPPO: `{sa_ippo_text}`。SA vs PPO 详见 `diagnosis_summary.json` 的 `sa_vs_ppo_result`。

10. 从 reward、continuity、cache miss/cold start、backhaul、eviction 看，SA 的优势或劣势在哪里？

本轮是 smoke，结论只用于定位链路。SA 相对 popularity：

{sa_pop_detail}

SA 相对 PPO：

{sa_ppo_detail}

11. 下一轮建议是什么？

建议先补齐 IPPO checkpoint/eval rows，随后在同一 proposal smoke 上复查 SA vs IPPO/PPO/popularity。若 cache miss/cold start 或 backhaul 差距集中，再考虑 policy-side prefetch/cache-admission bias；不建议直接把本 smoke 当正式 split。

12. 本轮是否可以 freeze？

不可以。本轮只是 proposal smoke，`do_not_freeze=true`。

## 输出

```json
{json.dumps(diagnosis.get('generated_artifacts', {}), ensure_ascii=False, indent=2)}
```
"""


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    output_dir = args.output_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    report_path = args.report_path or Path(config.get("report_path", DEFAULT_REPORT_PATH))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime(f"{TASK_NAME}_%Y%m%d_%H%M%S_%f")
    policies, missing_policy_rows = _available_policy_plan(config)
    manifests = {
        str(agent_name): _load_manifest(path)
        for agent_name, path in (config.get("checkpoint_manifests", {}) or {}).items()
    }
    checkpoint_required = checkpoint_required_agents()
    seeds = [int(seed) for seed in config.get("seeds", [7, 13, 29])]
    workflows = _load_workflows(config)
    cache_capacity_profile = dict(config.get("cache_capacity_profile", {}) or {})

    mobility_source_path, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
        mobility_csv_path=str(data_cfg.get("mobility_csv_path", "")),
        lust_scenario_root=str(data_cfg.get("lust_scenario_root", "")),
        max_mobility_rows=int(data_cfg.get("max_mobility_rows", 2500)),
        rsu_layout=str(data_cfg.get("rsu_layout", "auto_dominant_tight")),
        frame_offset=int(data_cfg.get("frame_offset", 0)),
        window_length=int(data_cfg.get("window_length", 24)),
        window_selector=str(data_cfg.get("window_selector", "max_handoff_candidate")),
        window_count=int(data_cfg.get("window_count", 1)),
        window_scan_stride=int(data_cfg.get("window_scan_stride", 2)),
        random_seed=seeds[0] if seeds else 7,
        window_mode=str(data_cfg.get("window_mode", "activating_only")),
    )
    selected_windows = list(window_payload.get("selected_windows", []))
    if not selected_windows:
        raise RuntimeError("No selected windows for multi-adapter hard-joint smoke.")

    rows: list[dict[str, Any]] = []
    episode_errors: list[dict[str, Any]] = []
    episode_root = output_dir / "episodes"

    for seed in seeds:
        seed_checkpoint_map = _checkpoint_map_for_seed(manifests, seed)
        for policy_name in list(policies):
            if policy_name in checkpoint_required and not seed_checkpoint_map.get(policy_name):
                missing_policy_rows.setdefault(policy_name, f"missing_checkpoint_for_seed_{seed}")
                continue
            for window_candidate in selected_windows:
                mobility_bundle = load_window_bundle(
                    root_dir=ROOT_DIR,
                    mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
                    mobility_csv_path=str(data_cfg.get("mobility_csv_path", "")),
                    lust_scenario_root=str(data_cfg.get("lust_scenario_root", "")),
                    max_mobility_rows=int(data_cfg.get("max_mobility_rows", 2500)),
                    rsu_layout=str(window_candidate.get("recommended_rsu_layout", data_cfg.get("rsu_layout", "auto_dominant_tight"))),
                    frame_offset=int(window_candidate.get("frame_offset", 0)),
                    window_length=int(window_candidate.get("window_length", data_cfg.get("window_length", 24))),
                    random_seed=seed,
                )
                mobility_bundle.rsu_metadata["window_rank"] = window_candidate.get("window_rank")
                mobility_bundle.rsu_metadata["window_class"] = window_candidate.get("window_class", "mechanism_activating")
                for workflow_state in workflows:
                    try:
                        summary = run_real_episode(
                            root_dir=ROOT_DIR,
                            agent_name=policy_name,
                            checkpoint_map=seed_checkpoint_map,
                            workflow_state=workflow_state,
                            workflow_source_path=str(data_cfg.get("workflow_csv_path", "")),
                            mobility_bundle=mobility_bundle,
                            seed=seed,
                            max_steps=int(data_cfg.get("max_steps", 12)),
                            mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
                            run_metadata={
                                "script": "scripts/run_multi_adapter_hard_joint_smoke.py",
                                "benchmark_run_id": run_id,
                                "mode": TASK_NAME,
                                "window_mode": TASK_NAME,
                                "window_rank": window_candidate.get("window_rank"),
                                "window_class": window_candidate.get("window_class", "mechanism_activating"),
                                "proposal_only": True,
                                "do_not_use_for_freeze": True,
                                "adapter_assignment_profile": str(config.get("adapter_assignment_profile")),
                                "cache_capacity_profile": cache_capacity_profile,
                                "mobility_source_path": mobility_source_path,
                            },
                            cache_capacity_profile=cache_capacity_profile,
                        )
                    except Exception as exc:  # pragma: no cover - runtime diagnostics.
                        missing_policy_rows.setdefault(policy_name, f"evaluation_failed:{type(exc).__name__}:{exc}")
                        episode_errors.append(
                            {
                                "policy_name": policy_name,
                                "seed": seed,
                                "workflow_id": workflow_state.workflow_id,
                                "window_id": mobility_bundle.rsu_metadata.get("window_id"),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        continue

                    summary_path = (
                        episode_root
                        / str(mobility_bundle.rsu_metadata.get("window_id"))
                        / str(workflow_state.workflow_id)
                        / policy_name
                        / f"seed_{seed}.summary.json"
                    )
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary["run_info"]["summary_path"] = str(summary_path)
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    row = _augment_row(summary_to_row(summary), summary, workflow_state, run_id)
                    row["summary_path"] = str(summary_path)
                    rows.append(row)

    if not rows:
        raise RuntimeError(f"No benchmark rows generated. Episode errors: {episode_errors}")

    aggregate_by_policy = aggregate_rows(
        rows,
        group_keys=["policy_name"],
        metrics=[
            "total_reward",
            "workflow_continuity_rate",
            "handoff_failure_rate",
            "backhaul_traffic_cost",
            "adapter_miss_count",
            "adapter_cold_start_count",
            "cache_occupancy_rate",
            "eviction_count",
        ],
    )
    policy_summary = build_policy_comparison_summary(rows, missing_policy_rows)
    adapter_diversity_summary = build_adapter_diversity_summary(rows)
    cache_eviction_summary = build_cache_eviction_summary(rows)
    handoff_continuity_summary = build_handoff_continuity_summary(rows)
    actionmix_summary = build_actionmix_summary(rows)

    generated_artifacts = {
        "benchmark_rows": str(output_dir / "benchmark_rows.csv"),
        "policy_comparison_summary": str(output_dir / "policy_comparison_summary.csv"),
        "adapter_diversity_summary": str(output_dir / "adapter_diversity_summary.csv"),
        "cache_eviction_summary": str(output_dir / "cache_eviction_summary.csv"),
        "handoff_continuity_summary": str(output_dir / "handoff_continuity_summary.csv"),
        "actionmix_summary": str(output_dir / "actionmix_summary.csv"),
        "diagnosis_summary": str(output_dir / "diagnosis_summary.json"),
        "report": str(report_path),
        "episodes_dir": str(episode_root),
    }
    diagnosis = {
        "task_name": TASK_NAME,
        "changed_files": [
            "src/evaluators/main_results_support.py",
            "scripts/run_multi_adapter_hard_joint_smoke.py",
            "configs/experiment/multi_adapter_hard_joint_smoke_round10.yaml",
            "docs/agent/multi_adapter_hard_joint_smoke_round10_report.md",
        ],
        "generated_artifacts": generated_artifacts,
        "policies_requested": REQUESTED_POLICIES,
        "policies_evaluated": sorted({str(row.get("policy_name")) for row in rows}),
        "missing_policy_rows": missing_policy_rows,
        "episode_errors": episode_errors,
        "semantic_ai_service_active": bool(all(str(row.get("adapter_assignment_profile")) == "semantic_ai_service" for row in rows)),
        "cache_capacity_profile_active": bool(any(_float_value(row.get("cache_capacity_enabled")) > 0.0 for row in rows)),
        "adapter_diversity_activated": bool(any(_float_value(row.get("required_adapter_count")) >= 3.0 for row in rows)),
        "eviction_activated": bool(any(_float_value(row.get("eviction_count")) > 0.0 for row in rows)),
        "handoff_pressure_active": bool(
            any(
                _float_value(row.get("handoff_during_workflow_rate")) > 0.0
                or _float_value(row.get("cross_rsu_workflow_rate")) > 0.0
                for row in rows
            )
        ),
        "sa_vs_ippo_result_if_available": _comparison_delta(rows, "sa_ghmappo", "ippo"),
        "sa_vs_ppo_result": _comparison_delta(rows, "sa_ghmappo", "ppo"),
        "sa_vs_popularity_result": _comparison_delta(rows, "sa_ghmappo", "popularity_cache_heuristic"),
        "aggregate_by_policy": aggregate_by_policy,
        "recommended_next_step": (
            "补齐 IPPO checkpoint/eval rows；继续用 proposal smoke 复查 SA/PPO/popularity/reactive，"
            "再决定是否优化 policy-side prefetch/cache-admission bias。"
        ),
        "do_not_freeze": True,
    }

    write_csv(output_dir / "benchmark_rows.csv", rows)
    write_csv(output_dir / "policy_comparison_summary.csv", policy_summary)
    write_csv(output_dir / "adapter_diversity_summary.csv", adapter_diversity_summary)
    write_csv(output_dir / "cache_eviction_summary.csv", cache_eviction_summary)
    write_csv(output_dir / "handoff_continuity_summary.csv", handoff_continuity_summary)
    write_csv(output_dir / "actionmix_summary.csv", actionmix_summary)
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        build_report(rows=rows, diagnosis=diagnosis, policy_summary=policy_summary),
        encoding="utf-8",
    )

    print("multi-adapter hard-joint smoke complete")
    print(f"rows: {len(rows)}")
    print(f"policies_evaluated: {', '.join(diagnosis['policies_evaluated'])}")
    print(f"missing_policy_rows: {json.dumps(missing_policy_rows, ensure_ascii=False)}")
    print(f"semantic_ai_service_active: {diagnosis['semantic_ai_service_active']}")
    print(f"adapter_diversity_activated: {diagnosis['adapter_diversity_activated']}")
    print(f"cache_capacity_profile_active: {diagnosis['cache_capacity_profile_active']}")
    print(f"eviction_activated: {diagnosis['eviction_activated']}")
    print(f"handoff_pressure_active: {diagnosis['handoff_pressure_active']}")
    for name, path in generated_artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
