"""Round11 project state, baseline inventory, IPPO/PPO readiness, and continuity audit."""

from __future__ import annotations

import csv
import inspect
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import ALGO_REGISTRY, checkpoint_required_agents, list_evaluable_agents


TASK_NAME = "project_state_sync_and_baseline_inventory_round11"
OUTPUT_DIR = Path("artifacts/analysis/project_state_sync_and_baseline_inventory_round11")
ROUND10_DIR = Path("artifacts/analysis/multi_adapter_hard_joint_smoke_round10")
PROJECT_STATE_DOC = Path("docs/agent/project_current_state_round11.md")
REPORT_PATH = Path("docs/agent/project_state_sync_and_baseline_inventory_round11_report.md")

ROUND5_FACTS = {
    "mixed_sa_reward": 83.405000,
    "mixed_popularity_reward": 83.513333,
    "mixed_delta": -0.108333,
    "full_sa_reward": 76.654815,
    "full_popularity_reward": 75.492778,
    "full_delta": 1.162037,
    "mechanism_activating_mean_delta": -0.162500,
    "prefetch_attempt_delta": -1.083333,
    "migration_attempt_delta": 1.083333,
    "backhaul_delta": -69.333333,
    "continuity_reward_component_delta": -0.162500,
}

ROUND10_REQUESTED_POLICIES = [
    "sa_ghmappo",
    "ippo",
    "ppo",
    "popularity_cache_heuristic",
    "reactive_greedy",
]
EXPLICIT_INVENTORY_CANDIDATES = [
    "sa_ghmappo",
    "ppo",
    "flat_ppo",
    "ppo_real",
    "ippo",
    "mappo",
    "flat_mappo",
    "popularity_cache_heuristic",
    "reactive_greedy",
    "reactive_offloading",
    "reactive_caching",
    "local_only",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "missing"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"


def _find_config_files(policy_name: str) -> list[str]:
    hits: list[str] = []
    search_terms = {policy_name}
    if policy_name == "ppo":
        search_terms.update({"flat_ppo", "ppo_real"})
    if policy_name == "flat_ppo":
        search_terms.update({"ppo"})
    for path in Path("configs").rglob("*"):
        if not path.is_file():
            continue
        lowered_name = path.name.lower()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            text = ""
        if any(term.lower() in lowered_name or term.lower() in text for term in search_terms):
            hits.append(str(path))
    return sorted(set(hits))[:6]


def _find_checkpoint_hits(policy_name: str) -> list[str]:
    terms = {policy_name}
    if policy_name == "ppo":
        terms.update({"flat_ppo", "ppo_real"})
    if policy_name == "flat_ppo":
        terms.update({"ppo"})
    if policy_name == "ippo":
        terms.update({"ippo"})
    hits: list[str] = []
    artifact_root = Path("artifacts")
    if not artifact_root.exists():
        return hits
    for path in artifact_root.rglob("*.pt"):
        path_text = str(path).lower()
        if any(term.lower() in path_text for term in terms):
            hits.append(str(path))
            if len(hits) >= 6:
                break
    return hits


def _find_artifact_mentions(policy_name: str) -> list[str]:
    terms = {policy_name}
    if policy_name == "ppo":
        terms.update({"flat_ppo", "ppo_real"})
    hits: list[str] = []
    roots = [Path("artifacts/paper"), Path("artifacts/benchmarks"), Path("docs/agent")]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
                continue
            name_hit = any(term.lower() in path.name.lower() for term in terms)
            text_hit = False
            if not name_hit:
                try:
                    head = path.read_text(encoding="utf-8", errors="ignore")[:200000].lower()
                except Exception:
                    head = ""
                text_hit = any(term.lower() in head for term in terms)
            if name_hit or text_hit:
                hits.append(str(path))
                if len(hits) >= 6:
                    return hits
    return hits


def _source_file_for_registry_policy(policy_name: str) -> str:
    spec = ALGO_REGISTRY.get(policy_name)
    if not spec:
        return "missing"
    agent_class = spec.get("class")
    try:
        source = inspect.getsourcefile(agent_class)
    except Exception:
        source = None
    return str(Path(source).relative_to(ROOT_DIR)) if source else "unknown"


def _category_for(policy_name: str, registered: bool, source_file: str, artifact_mentions: list[str]) -> str:
    if policy_name == "sa_ghmappo":
        return "main_method"
    if policy_name in {"ppo", "flat_ppo", "mappo", "flat_mappo"}:
        return "rl_baseline"
    if policy_name in {"ippo", "ppo_real"}:
        return "legacy" if artifact_mentions and not registered else "rl_baseline"
    if policy_name in {"popularity_cache_heuristic", "reactive_greedy"}:
        return "heuristic_baseline"
    if policy_name in {"reactive_offloading", "reactive_caching", "local_only"}:
        return "hand_written_rule"
    return "unknown"


def _round10_policy_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("policy_name") or row.get("agent_name")) for row in rows if row.get("policy_name") or row.get("agent_name")}


def build_policy_inventory(round10_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluable = set(list_evaluable_agents())
    checkpoint_required = checkpoint_required_agents()
    candidates = set(EXPLICIT_INVENTORY_CANDIDATES)
    candidates.update(ALGO_REGISTRY.keys())
    candidates.update(_round10_policy_names(round10_rows))
    round10_policies = _round10_policy_names(round10_rows)

    rows: list[dict[str, Any]] = []
    for policy_name in sorted(candidates):
        registered = policy_name in ALGO_REGISTRY
        source_file = _source_file_for_registry_policy(policy_name) if registered else "missing"
        artifact_mentions = _find_artifact_mentions(policy_name)
        config_files = _find_config_files(policy_name)
        checkpoint_hits = _find_checkpoint_hits(policy_name)
        eval_available = policy_name in evaluable
        checkpoint_req: bool | None
        if registered:
            checkpoint_req = policy_name in checkpoint_required
        elif policy_name in {"ippo", "ppo_real"} and artifact_mentions:
            checkpoint_req = True
        else:
            checkpoint_req = None
        appeared = policy_name in round10_policies

        if appeared:
            missing_reason = "none"
            recommended_action = "keep_in_smoke_and_formal_audit"
        elif not registered and policy_name == "ippo":
            missing_reason = "not_registered_or_not_evaluable_in_live_registry"
            recommended_action = "restore_or_add_minimal_live_ippo_eval_contract_then_use_existing_or_new_checkpoint"
        elif not registered and policy_name == "ppo_real":
            missing_reason = "legacy_artifact_name_not_live_registry; current ppo maps to PPOAgent/flat_ppo checkpoint alias"
            recommended_action = "do_not_add_new_algorithm_name_unless needed; document ppo/flat_ppo alias"
        elif not registered and policy_name in {"reactive_offloading", "reactive_caching", "local_only"}:
            missing_reason = "not_detected_as_live_agent_or_runner_choice"
            recommended_action = "only add rows after locating or implementing explicit live rule contract"
        elif registered and policy_name not in evaluable:
            missing_reason = "registered_but_not_evaluable"
            recommended_action = "inspect registry support_level before smoke"
        elif registered and policy_name in checkpoint_required and not checkpoint_hits:
            missing_reason = "checkpoint_required_but_no_checkpoint_found"
            recommended_action = "run smoke training or provide checkpoint before eval"
        else:
            missing_reason = "not_selected_for_round10"
            recommended_action = "include only if needed for next comparison scope"

        rows.append(
            {
                "policy_name": policy_name,
                "category": _category_for(policy_name, registered, source_file, artifact_mentions),
                "source_file": source_file,
                "config_file_if_any": ";".join(config_files) if config_files else "missing",
                "eval_entry_available": _bool_text(eval_available),
                "checkpoint_required": _bool_text(checkpoint_req),
                "checkpoint_found": _bool_text(bool(checkpoint_hits)) if checkpoint_req is not None else "unknown",
                "checkpoint_examples": ";".join(checkpoint_hits) if checkpoint_hits else "missing",
                "registered_in_runner": _bool_text(eval_available),
                "appeared_in_round10_rows": _bool_text(appeared),
                "artifact_mentions": ";".join(artifact_mentions) if artifact_mentions else "missing",
                "missing_reason": missing_reason,
                "recommended_action": recommended_action,
            }
        )
    return rows


def build_ippo_ppo_readiness(inventory_rows: list[dict[str, Any]], round10_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy = {row["policy_name"]: row for row in inventory_rows}
    ppo_rows = [row for row in round10_rows if str(row.get("policy_name")) == "ppo"]
    ppo_checkpoint_run_ids = sorted({str(row.get("checkpoint_run_id")) for row in ppo_rows if row.get("checkpoint_run_id")})
    ppo_from_flat_alias = bool(any("flat_ppo" in item for item in ppo_checkpoint_run_ids))
    ippo_inventory = by_policy.get("ippo", {})
    ppo_inventory = by_policy.get("ppo", {})
    ppo_real_inventory = by_policy.get("ppo_real", {})
    return [
        {
            "policy_name": "ippo",
            "agent_file_exists": "false",
            "config_exists": _bool_text(Path("configs/algo/ippo.yaml").exists()),
            "config_note": "No dedicated configs/algo/ippo.yaml was found; proposal configs may mention ippo as requested policy only.",
            "checkpoint_found": ippo_inventory.get("checkpoint_found", "unknown"),
            "eval_runner_registered": ippo_inventory.get("eval_entry_available", "unknown"),
            "round10_status": "missing",
            "round10_missing_reason": "not_registered_or_not_evaluable",
            "readiness_status": "not_ready_live_registry_missing",
            "answer": "IPPO has historical artifacts/old benchmark rows, but no current src/agents file and no live registry entry.",
            "minimum_change_to_add_rows": "add or restore an IPPO agent/eval registry contract and point runner to a compatible checkpoint; if no compatible checkpoint is accepted, run short IPPO smoke training first",
            "need_training": "unknown_until_live_contract_checked",
            "why_checkpoint_not_found_by_runner": "runner only evaluates list_evaluable_agents; ippo is absent from current registry",
        },
        {
            "policy_name": "ppo",
            "agent_file_exists": "true",
            "config_exists": _bool_text(Path("configs/algo/ppo.yaml").exists()),
            "config_note": "Current live PPO config exists and flat_ppo is treated as legacy alias.",
            "checkpoint_found": ppo_inventory.get("checkpoint_found", "unknown"),
            "eval_runner_registered": ppo_inventory.get("eval_entry_available", "unknown"),
            "round10_status": "evaluated",
            "round10_checkpoint_run_ids": ";".join(ppo_checkpoint_run_ids) if ppo_checkpoint_run_ids else "missing",
            "round10_uses_flat_ppo_alias": _bool_text(ppo_from_flat_alias),
            "readiness_status": "ready",
            "answer": "Current ppo rows are PPOAgent rows loaded from existing flat_ppo checkpoint aliases.",
            "minimum_change_to_add_rows": "none for ppo; keep alias documented",
            "need_training": "false_for_round10_smoke",
            "why_checkpoint_not_found_by_runner": "not_applicable",
        },
        {
            "policy_name": "ppo_real",
            "agent_file_exists": "false",
            "config_exists": _bool_text(Path("configs/algo/ppo_real.yaml").exists()),
            "config_note": "No dedicated live ppo_real config was found; ppo_real is a historical artifact label.",
            "checkpoint_found": ppo_real_inventory.get("checkpoint_found", "unknown"),
            "eval_runner_registered": ppo_real_inventory.get("eval_entry_available", "unknown"),
            "round10_status": "not_requested",
            "round10_missing_reason": "legacy_name_not_live_registry",
            "readiness_status": "legacy_artifact_name",
            "answer": "ppo_real appears in historical paper artifacts, but current live registry exposes ppo/flat_ppo instead.",
            "minimum_change_to_add_rows": "prefer ppo/flat_ppo alias unless a strict ppo_real legacy reproduction is required",
            "need_training": "false_if_ppo_alias_is_accepted",
            "why_checkpoint_not_found_by_runner": "not requested and not registered as live agent",
        },
    ]


def build_continuity_audit(round10_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sa_rows = [row for row in round10_rows if str(row.get("policy_name")) == "sa_ghmappo"]
    pop_rows = [row for row in round10_rows if str(row.get("policy_name")) == "popularity_cache_heuristic"]
    sa_continuity = _mean([_float_value(row.get("workflow_continuity_rate")) for row in sa_rows])
    sa_failure = _mean([_float_value(row.get("handoff_failure_rate")) for row in sa_rows])
    sa_success = _mean([_float_value(row.get("successful_episode_rate", row.get("success"))) for row in sa_rows])
    sa_miss = _mean([_float_value(row.get("adapter_miss_count")) for row in sa_rows])
    pop_continuity = _mean([_float_value(row.get("workflow_continuity_rate")) for row in pop_rows])
    pop_backhaul = _mean([_float_value(row.get("backhaul_traffic_cost")) for row in pop_rows])
    sa_backhaul = _mean([_float_value(row.get("backhaul_traffic_cost")) for row in sa_rows])
    return [
        {
            "continuity_field_name": "workflow_continuity_rate",
            "source_file": "src/metrics/paper_metrics.py;src/metrics/recorder.py;src/evaluators/main_results_support.py",
            "calculation_location": "PaperMetricSet.compute -> summary_to_row",
            "numerator": "count(step_records where stall_occurred == false)",
            "denominator": "total step_records in episode",
            "aggregation_method": "episode_step_ratio_then_policy_mean_in_round10_summary",
            "whether_continuity_equals_success_rate": "false",
            "whether_continuity_can_be_below_1_while_failure_0": "true",
            "round10_sa_mean": sa_continuity,
            "round10_popularity_mean": pop_continuity,
            "round10_sa_failure_mean": sa_failure,
            "round10_sa_success_rate_mean": sa_success,
            "explanation": (
                "This is a per-step non-stall ratio, not workflow completion success. "
                "A row can have no failed handoff but still stall on cache miss, missing base model, or invalid offload target."
            ),
            "related_fields": (
                "stall_occurred;workflow_continuity_rate;handoff_failure_rate;handoff_ready_ratio;"
                "adapter_miss_count;adapter_cold_start_count;prefetch_success_count;migration_success_count"
            ),
        },
        {
            "continuity_field_name": "continuity_reward_component",
            "source_file": "src/envs/core/vec_workflow_core_env.py;src/evaluators/main_results_support.py",
            "calculation_location": "RewardBreakdown.continuity_bonus summed by _build_actionmix_diagnostics",
            "numerator": "sum(reward_dict['continuity_bonus'] over step_trace)",
            "denominator": "none",
            "aggregation_method": "reward_component_sum_per_episode_then_policy_mean",
            "whether_continuity_equals_success_rate": "false",
            "whether_continuity_can_be_below_1_while_failure_0": "not_applicable",
            "round10_sa_mean": _mean([_float_value(row.get("continuity_reward_component")) for row in sa_rows]),
            "round10_popularity_mean": _mean([_float_value(row.get("continuity_reward_component")) for row in pop_rows]),
            "round10_sa_adapter_miss_mean": sa_miss,
            "round10_sa_backhaul_mean": sa_backhaul,
            "round10_popularity_backhaul_mean": pop_backhaul,
            "explanation": (
                "This is a reward proxy, not the metric ratio. In round10 SA has lower step continuity than popularity, "
                "but also lower backhaul and higher continuity_reward_component mean, so total reward can still be higher."
            ),
            "related_fields": "continuity_bonus;cache_hit;warm_ready;handoff_ready;migration_prepare;backhaul_traffic_cost",
        },
        {
            "continuity_field_name": "handoff_failure_rate",
            "source_file": "src/metrics/paper_metrics.py;src/metrics/recorder.py",
            "calculation_location": "PaperMetricSet.compute",
            "numerator": "count(step_records where handoff_failed == true)",
            "denominator": "sum(handoff_event_count)",
            "aggregation_method": "episode_handoff_failure_ratio_then_policy_mean",
            "whether_continuity_equals_success_rate": "false",
            "whether_continuity_can_be_below_1_while_failure_0": "true",
            "round10_sa_mean": sa_failure,
            "round10_popularity_mean": _mean([_float_value(row.get("handoff_failure_rate")) for row in pop_rows]),
            "explanation": (
                "Failure=0 only says no handoff event was counted as failed. It does not imply every service step avoided stall."
            ),
            "related_fields": "handoff_failed;handoff_event_count;handoff_ready;workflow_continuity_rate",
        },
    ]


def _load_round10_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    return (
        _read_csv(ROUND10_DIR / "benchmark_rows.csv"),
        _read_csv(ROUND10_DIR / "policy_comparison_summary.csv"),
        _read_json(ROUND10_DIR / "diagnosis_summary.json"),
    )


def _policy_summary_map(policy_summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("policy_name")): row for row in policy_summary_rows}


def build_project_state_summary(
    *,
    inventory_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    continuity_rows: list[dict[str, Any]],
    round10_summary_rows: list[dict[str, Any]],
    round10_diagnosis: dict[str, Any],
) -> dict[str, Any]:
    policy_map = _policy_summary_map(round10_summary_rows)
    return {
        "task_name": TASK_NAME,
        "training_run": False,
        "freeze_run": False,
        "reward_modified": False,
        "policy_modified": False,
        "baseline_modified": False,
        "checkpoint_selection_modified": False,
        "current_main_method": "sa_ghmappo",
        "benchmark_layers": {
            "formal_candidate_layers": ["mixed_informative", "full_stratified"],
            "proposal_smoke_layers": ["multi_adapter_hard_joint_proposal", "multi_adapter_hard_joint_smoke_round10"],
        },
        "round5_facts": ROUND5_FACTS,
        "round10": {
            "do_not_freeze": True,
            "rows": len(_read_csv(ROUND10_DIR / "benchmark_rows.csv")),
            "policies_evaluated": round10_diagnosis.get("policies_evaluated", []),
            "missing_policy_rows": round10_diagnosis.get("missing_policy_rows", {}),
            "semantic_ai_service_active": round10_diagnosis.get("semantic_ai_service_active"),
            "adapter_diversity_activated": round10_diagnosis.get("adapter_diversity_activated"),
            "cache_capacity_profile_active": round10_diagnosis.get("cache_capacity_profile_active"),
            "eviction_activated": round10_diagnosis.get("eviction_activated"),
            "handoff_pressure_active": round10_diagnosis.get("handoff_pressure_active"),
            "policy_summary": policy_map,
        },
        "inventory_counts": {
            "total_policies": len(inventory_rows),
            "round10_evaluated": sum(1 for row in inventory_rows if row.get("appeared_in_round10_rows") == "true"),
            "heuristic_or_handwritten": sum(
                1 for row in inventory_rows if row.get("category") in {"heuristic_baseline", "hand_written_rule"}
            ),
        },
        "ippo_readiness": next((row for row in readiness_rows if row.get("policy_name") == "ippo"), {}),
        "ppo_readiness": next((row for row in readiness_rows if row.get("policy_name") == "ppo"), {}),
        "continuity_audit": continuity_rows,
        "recommended_next_priority": [
            "补 IPPO live eval/checkpoint rows",
            "确认是否存在遗漏 hand-written rule rows；当前 live heuristics 是 popularity_cache_heuristic 与 reactive_greedy",
            "做 hard_joint_policy_failure_diagnosis",
            "最后再考虑 policy-side prefetch/cache-admission bias",
        ],
    }


def build_project_state_doc(summary: dict[str, Any]) -> str:
    round10 = summary["round10"]
    policy_summary = round10.get("policy_summary", {})
    return f"""# project_current_state_round11

## 当前主线目标

当前项目主线仍是 `NGSIM + Alibaba`，研究对象是 AI-driven VEC 中跨 RSU 连续 DAG workflow 执行、adapter cache 协同、handoff 状态迁移与多时间尺度控制。

当前主算法是 `sa_ghmappo`。当前不应继续盲调 policy；下一步应先补齐对照方法 rows 与指标定义审计。

## Benchmark 层级

- `mixed_informative`：正式候选层，用于当前 round2 qualified candidate 的 mixed 对照。
- `full_stratified`：正式候选层，用于当前 round2 qualified candidate 的 full 对照。
- `multi_adapter_hard_joint_proposal`：proposal only，不是新真实数据集，不可 freeze。
- `multi_adapter_hard_joint_smoke_round10`：proposal smoke，不是正式论文结果，不可 freeze。

明确：`multi_adapter_hard_joint_smoke_round10` 不是新数据集，不是正式论文结果，不可 freeze。它只是在真实 mobility trace + 真实 Alibaba DAG structure 上叠加 controlled AI-service adapter assignment 与 controlled cache-capacity stress。

## round5-round10 状态摘要

- round5：mixed 中 SA reward `83.405000`，popularity `83.513333`，SA-pop `-0.108333`；full 中 SA reward `76.654815`，popularity `75.492778`，SA-pop `+1.162037`。mixed gap 集中在 `mechanism_activating`，SA 少 prefetch、多 migration prepare、backhaul 更低，剩余 gap 主要表现为 continuity reward tie-break。
- scenario innovation audit：当前数据不是 easy static；`hard_joint=0.666667`，`mechanism_activating=0.333333`。`j_3` 是 9 nodes / critical path 5，`j_8` 是 17 nodes / critical path 9。handoff/cross-RSU pressure 存在，但 adapter diversity 只有 1。
- round7：catalog 中有 5 个 adapter，但 benchmark 只出现 `adapter_batch_type_1`；原因是 j_3/j_8 的 Alibaba `task_type=1` 被 legacy parser 映射到单一 adapter。base model 只有 `veh_base_v1`，且当时没有 capacity/eviction。
- round8：新增 `adapter_assignment_profile`，默认 `legacy_batch_type` 不变；`semantic_ai_service` 只在显式启用时生效，j_3 产生 4 个 adapter，j_8 产生 5 个 adapter，全部在 catalog 中。
- round9：新增默认关闭的 cache capacity/eviction telemetry。`enabled=false` 保持 append-only；`enabled=true, rsu_adapter_slots=2` 时 LRU eviction 生效。
- round10：生成 `{round10.get('rows')}` 条 proposal smoke rows，`semantic_ai_service_active={round10.get('semantic_ai_service_active')}`，`adapter_diversity_activated={round10.get('adapter_diversity_activated')}`，`cache_capacity_profile_active={round10.get('cache_capacity_profile_active')}`，`eviction_activated={round10.get('eviction_activated')}`，`handoff_pressure_active={round10.get('handoff_pressure_active')}`。

## round10 policy 状态

- `sa_ghmappo`: reward `{policy_summary.get('sa_ghmappo', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('sa_ghmappo', {}).get('workflow_continuity_rate_mean')}`, failure `{policy_summary.get('sa_ghmappo', {}).get('handoff_failure_rate_mean')}`, backhaul `{policy_summary.get('sa_ghmappo', {}).get('backhaul_traffic_cost_mean')}`。
- `ppo`: reward `{policy_summary.get('ppo', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('ppo', {}).get('workflow_continuity_rate_mean')}`；row 来自 current `ppo` registry + existing `flat_ppo` checkpoint alias。
- `popularity_cache_heuristic`: reward `{policy_summary.get('popularity_cache_heuristic', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('popularity_cache_heuristic', {}).get('workflow_continuity_rate_mean')}`。
- `reactive_greedy`: reward `{policy_summary.get('reactive_greedy', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('reactive_greedy', {}).get('workflow_continuity_rate_mean')}`。
- `ippo`: missing，原因是 `not_registered_or_not_evaluable`。

## 为什么现在不能 freeze

- round10 是 proposal smoke，不是正式 benchmark split。
- IPPO rows 缺失，SA vs IPPO 仍无法回答。
- continuity 指标在 hard joint smoke 下暴露出 SA 相对 popularity 的短板，需要先做 failure diagnosis，而不是直接 freeze。
- controlled AI-service/cache stress 不应被写成真实数据集或正式论文结论。

## 下一步优先级

1. 先补 IPPO live eval/checkpoint rows。
2. 确认是否存在遗漏 hand-written rule rows；当前 live heuristic rows 是 `popularity_cache_heuristic` 与 `reactive_greedy`。
3. 做 `hard_joint_policy_failure_diagnosis`，定位 SA continuity 低但 reward 高的具体 step/window/workflow 原因。
4. 最后才考虑 policy-side prefetch/cache-admission bias。
"""


def build_report(summary: dict[str, Any], inventory_rows: list[dict[str, Any]]) -> str:
    round10 = summary["round10"]
    policy_summary = round10.get("policy_summary", {})
    missing = round10.get("missing_policy_rows", {})
    heuristic_rows = [
        row for row in inventory_rows if row.get("category") in {"heuristic_baseline", "hand_written_rule"}
    ]
    missing_handwritten = [
        row for row in heuristic_rows if row.get("appeared_in_round10_rows") != "true"
    ]
    continuity_row = next(row for row in summary["continuity_audit"] if row["continuity_field_name"] == "workflow_continuity_rate")
    return f"""# project_state_sync_and_baseline_inventory_round11 报告

## 结论

本轮只做项目状态同步、baseline inventory、IPPO/PPO readiness 和 continuity 指标定义审计。没有训练、没有 freeze、没有修改 reward、policy、baseline、checkpoint selection 或正式 benchmark split。

## 1. 当前项目最新状态是什么？

主线仍是 `NGSIM + Alibaba`。主方法是 `sa_ghmappo`。正式候选结果仍应看 `mixed_informative` 与 `full_stratified`；`multi_adapter_hard_joint_proposal/smoke` 只是 proposal smoke。

## 2. round10 的结果是否可以 freeze？

不可以。round10 是 controlled AI-service/cache-stress smoke，不是新数据集，不是正式论文结果，不可 freeze。

## 3. SA-GHMAPPO 在 round10 smoke 中的表现

- SA: reward `{policy_summary.get('sa_ghmappo', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('sa_ghmappo', {}).get('workflow_continuity_rate_mean')}`, failure `{policy_summary.get('sa_ghmappo', {}).get('handoff_failure_rate_mean')}`, backhaul `{policy_summary.get('sa_ghmappo', {}).get('backhaul_traffic_cost_mean')}`。
- PPO: reward `{policy_summary.get('ppo', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('ppo', {}).get('workflow_continuity_rate_mean')}`, failure `{policy_summary.get('ppo', {}).get('handoff_failure_rate_mean')}`。
- Popularity: reward `{policy_summary.get('popularity_cache_heuristic', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('popularity_cache_heuristic', {}).get('workflow_continuity_rate_mean')}`, backhaul `{policy_summary.get('popularity_cache_heuristic', {}).get('backhaul_traffic_cost_mean')}`。
- Reactive: reward `{policy_summary.get('reactive_greedy', {}).get('total_reward_mean')}`, continuity `{policy_summary.get('reactive_greedy', {}).get('workflow_continuity_rate_mean')}`, failure `{policy_summary.get('reactive_greedy', {}).get('handoff_failure_rate_mean')}`。

SA 相对 PPO 在 reward、continuity、failure、backhaul、miss、eviction 上都更好。SA 相对 popularity reward 更高、backhaul 更低、eviction 更少，但 continuity 更低、adapter miss 更多。

## 4. IPPO 为什么缺失？

`ippo` 缺失原因是 `{missing.get('ippo', 'unknown')}`。当前 live `src/agents/registry.py` 没有 IPPO 条目，`list_evaluable_agents()` 不包含 IPPO。虽然历史 paper artifacts 中存在 `ippo` 行和 checkpoint 痕迹，但当前 live eval runner 不会评估它。

## 5. 手写规则 / heuristic baselines 有哪些？

当前 live heuristic baseline 是：

- `popularity_cache_heuristic`
- `reactive_greedy`

inventory 中额外检查了 `reactive_offloading`、`reactive_caching`、`local_only`，当前未发现 live agent/registry 入口。

## 6. 哪些手写规则没有进入 round10？

{chr(10).join(f"- `{row['policy_name']}`: {row['missing_reason']}" for row in missing_handwritten) if missing_handwritten else "- none"}

## 7. popularity_cache_heuristic 是否是当前最强 hand-written rule？

是，从当前 live heuristic 集合看，`popularity_cache_heuristic` 是最强的手写/规则基线：round10 reward `88.395000` 高于 `reactive_greedy` 的 `65.175000`，且 continuity 为 `1.0`、failure 为 `0.0`。

## 8. PPO row 是否就是 flat PPO / ppo_real alias？

round10 的 `ppo` row 来自 current `PPOAgent`，checkpoint_run_id 包含 `flat_ppo_train_...`，因此是 live `ppo` 名称加载 existing `flat_ppo` checkpoint alias。`ppo_real` 是历史 artifact 名称，不是当前 live registry 名称。

## 9. continuity=0.583334 是什么指标？

`workflow_continuity_rate = count(non-stall steps) / total step_records`。它来自 `src/metrics/paper_metrics.py::PaperMetricSet.compute`，再由 `summary_to_row()` 写入 benchmark rows。round10 SA 均值是 `{continuity_row['round10_sa_mean']}`。

## 10. failure=0 但 continuity 低是否矛盾？

不矛盾。`handoff_failure_rate` 的分子是 handoff failed events，分母是 handoff events；`workflow_continuity_rate` 的分子是非 stall step。没有 handoff failure 仍可能因为 cache miss、offload target、base model mismatch 等原因产生 service stall。

## 11. SA continuity 低的最可能原因是什么？

round10 中 SA adapter miss 均值为 `{policy_summary.get('sa_ghmappo', {}).get('adapter_miss_count_mean')}`，popularity 为 `{policy_summary.get('popularity_cache_heuristic', {}).get('adapter_miss_count_mean')}`。最可能原因是 proposal hard-joint 下 SA cache admission/prefetch 覆盖不足造成 step-level stall，而不是 handoff failure。

## 12. 下一轮优先级

1. 先补 IPPO eval/checkpoint rows。
2. 补遗漏 hand-written rule rows；当前没有 live `local_only/reactive_offloading/reactive_caching`。
3. 再做 `hard_joint_policy_failure_diagnosis`。
4. 最后才考虑 policy-side prefetch/cache-admission bias。

## 13. 是否建议现在继续调 SA policy？

不建议。当前应先补齐对照方法与指标定义审计，尤其是 IPPO live rows 和 hard-joint continuity/miss 的 step-level 诊断。

## 输出文件

- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/policy_baseline_inventory.csv`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/ippo_ppo_readiness.csv`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/continuity_metric_audit.csv`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/project_state_summary.json`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/diagnosis_summary.json`
- `docs/agent/project_current_state_round11.md`
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_STATE_DOC.parent.mkdir(parents=True, exist_ok=True)
    round10_rows, round10_summary_rows, round10_diagnosis = _load_round10_state()
    inventory_rows = build_policy_inventory(round10_rows)
    readiness_rows = build_ippo_ppo_readiness(inventory_rows, round10_rows)
    continuity_rows = build_continuity_audit(round10_rows)
    project_summary = build_project_state_summary(
        inventory_rows=inventory_rows,
        readiness_rows=readiness_rows,
        continuity_rows=continuity_rows,
        round10_summary_rows=round10_summary_rows,
        round10_diagnosis=round10_diagnosis,
    )
    diagnosis = {
        "task_name": TASK_NAME,
        "changed_files": [
            "scripts/audit_policy_baseline_inventory.py",
            "docs/agent/project_current_state_round11.md",
            "docs/agent/project_state_sync_and_baseline_inventory_round11_report.md",
        ],
        "generated_artifacts": {
            "policy_baseline_inventory": str(OUTPUT_DIR / "policy_baseline_inventory.csv"),
            "ippo_ppo_readiness": str(OUTPUT_DIR / "ippo_ppo_readiness.csv"),
            "continuity_metric_audit": str(OUTPUT_DIR / "continuity_metric_audit.csv"),
            "project_state_summary": str(OUTPUT_DIR / "project_state_summary.json"),
            "diagnosis_summary": str(OUTPUT_DIR / "diagnosis_summary.json"),
            "project_current_state_doc": str(PROJECT_STATE_DOC),
            "report": str(REPORT_PATH),
        },
        "training_run": False,
        "freeze_run": False,
        "reward_modified": False,
        "policy_modified": False,
        "baseline_modified": False,
        "checkpoint_selection_modified": False,
        "round10_can_freeze": False,
        "ippo_missing_reason": project_summary["round10"]["missing_policy_rows"].get("ippo", "unknown"),
        "ppo_round10_uses_flat_ppo_alias": next(
            (row for row in readiness_rows if row.get("policy_name") == "ppo"),
            {},
        ).get("round10_uses_flat_ppo_alias"),
        "live_heuristic_baselines": [
            row["policy_name"]
            for row in inventory_rows
            if row["category"] == "heuristic_baseline" and row["eval_entry_available"] == "true"
        ],
        "continuity_definition": "workflow_continuity_rate = non_stall_step_count / total_step_records",
        "recommended_next_step": project_summary["recommended_next_priority"],
    }

    _write_csv(OUTPUT_DIR / "policy_baseline_inventory.csv", inventory_rows)
    _write_csv(OUTPUT_DIR / "ippo_ppo_readiness.csv", readiness_rows)
    _write_csv(OUTPUT_DIR / "continuity_metric_audit.csv", continuity_rows)
    (OUTPUT_DIR / "project_state_summary.json").write_text(
        json.dumps(project_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    PROJECT_STATE_DOC.write_text(build_project_state_doc(project_summary), encoding="utf-8")
    REPORT_PATH.write_text(build_report(project_summary, inventory_rows), encoding="utf-8")

    print("project state sync and baseline inventory complete")
    print(f"inventory_rows: {len(inventory_rows)}")
    print(f"ippo_missing_reason: {diagnosis['ippo_missing_reason']}")
    print(f"ppo_round10_uses_flat_ppo_alias: {diagnosis['ppo_round10_uses_flat_ppo_alias']}")
    print(f"continuity_definition: {diagnosis['continuity_definition']}")
    for name, path in diagnosis["generated_artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
