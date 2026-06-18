"""Build top-journal comparison artifacts from a final-submission run.

The final-submission gate already decides whether the run is paper-ready.
This script turns that gate into reviewer-facing tables: protocol matrix,
main learned-baseline margins, mechanism-level paired statistics, support
suite summaries, and a compact markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


DEFAULT_FINAL_RUN_ROOT = Path(
    "artifacts/experiments/top_journal_final_submission/"
    "final_submission_controller_mappo_qmix_20260509_v1"
)
DEFAULT_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "mechanism_realization_rate",
]
LOWER_IS_BETTER = {
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
}
DEFAULT_LEARNED_BASELINES = [
    "ppo",
    "mappo",
    "dqn",
    "dueling_dqn",
    "qmix",
    "controller_mat",
    "dag_offload_drl",
    "cache_offload_drl",
    "dt_handoff_drl",
]
DEFAULT_HEURISTIC_REFERENCES = ["reactive_greedy", "popularity_cache_heuristic"]
PAPER_TABLE_AGENTS = [
    "sa_ghmappo",
    "ppo",
    "mappo",
    "dqn",
    "dueling_dqn",
    "qmix",
    "controller_mat",
    "dag_offload_drl",
    "cache_offload_drl",
    "dt_handoff_drl",
]
ACTION_MIX_AUDIT_AGENTS = PAPER_TABLE_AGENTS
ACTION_MIX_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "mechanism_realization_rate",
    "service_success_count",
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
    "prefetch_failed_count",
    "migration_attempt_count",
    "migration_success_count",
    "migration_failed_count",
    "mechanism_attempt_count",
    "mechanism_validated_success_count",
    "mechanism_success_rate",
    "mechanism_pending_success_count",
    "invalid_action_attempt_count",
    "env_invalid_action_count",
    "action_projection_count",
    "guard_action_delta_count",
    "dag_frontier_size_mean",
    "dag_critical_path_pressure_mean",
    "dag_current_node_dependency_pressure_mean",
    "dag_remaining_nodes_mean",
    "migration_overhead_sum",
    "mechanism_shaping_reward_component",
]
MAPPO_RISK_REFERENCE_AGENTS = ["ppo", "dqn"]
MAPPO_REWARD_RISK_DELTA = -5.0
MAPPO_CONTINUITY_RISK_DELTA = -0.2
AGENT_LABELS = {
    "sa_ghmappo": "SA-GHMAPPO",
    "ippo": "IPPO",
    "ppo": "PPO",
    "mappo": "MAPPO",
    "dqn": "DQN",
    "ddqn": "Double-DQN",
    "dueling_dqn": "Dueling-DQN",
    "dueling_ddqn": "Dueling-DDQN",
    "qmix": "QMIX",
    "controller_mat": "Controller-MAT",
    "dag_offload_drl": "DAG-Offload-DRL",
    "cache_offload_drl": "Cache-Offload-DRL",
    "dt_handoff_drl": "DT-Handoff-DRL",
    "reactive_greedy": "Reactive-Greedy",
    "popularity_cache_heuristic": "Popularity-Cache",
}
LATEX_ROLE_LABELS = {
    "candidate": "Main method",
    "primary_learned_comparator": "Primary learned",
    "not_in_current_primary_gate": "Available",
    "supplementary_reference": "Supplementary",
    "excluded": "Excluded",
}
LATEX_BUDGET_LABELS = {
    "matched_environment_interaction_budget": "Matched env. interactions",
    "matched_budget_but_protocol_blocked": "Matched but protocol-blocked",
    "same_budget_required_in_new_final_loop": "Same budget in new final loop",
    "same_budget_after_duplicate_trace_audit": "Same budget after audit",
    "not_trained": "Not trained",
}
LATEX_STATUS_LABELS = {
    "main_method": "Main",
    "primary_claimable": "Claimable",
    "primary_blocked_by_protocol": "Protocol-blocked",
    "available_needs_final_loop": "Needs final loop",
    "optional_needs_independence_audit": "Needs independence audit",
    "supplementary_only": "Supplementary",
    "excluded": "Excluded",
}
AUDITED_ROLE_LABELS = {
    "main_method": "Main method",
    "primary_learned_comparator": "Primary learned",
    "supplementary_reference": "Supplementary ref.",
}
AUDIT_DECISION_LABELS = {
    "pass_main_method": "Pass",
    "pass_primary_learned": "Pass",
    "pass_supplementary_reference": "Pass (supp.)",
}
SUITE_LABELS = {
    "formal_offset_0": "Formal",
    "holdout_offset_3": "Holdout",
}
MODE_LABELS = {
    "mixed_informative": "Mixed",
    "full_stratified": "Full",
}
SUPPORT_LABELS = {
    "prediction": "Prediction",
    "robustness": "Robustness",
    "scalability": "Scalability",
}
SUPPORT_ROW_FILES = {
    "prediction": ("prediction_robustness_rows.csv", "prediction_setting_id"),
    "robustness": ("benchmark_rows.csv", "robustness_setting_id"),
    "scalability": ("benchmark_rows.csv", "scalability_setting_id"),
}
CURRENT_MAPPO_PROTOCOL = {
    "head_credit_enabled": True,
    "head_credit_protocol": "aggregation_reason_weighted_controller_ppo_v3",
    "slow_policy_credit_floor": 0.25,
    "fast_policy_credit_floor": 0.10,
    "event_policy_credit_floor": 0.12,
    "slow_entropy_coef_scale": 1.25,
    "fast_entropy_coef_scale": 1.00,
    "event_entropy_coef_scale": 1.35,
    "slow_entropy_credit_floor": 0.20,
    "fast_entropy_credit_floor": 0.08,
    "event_entropy_credit_floor": 0.12,
    "event_advantage_blend": 0.85,
}
ALGORITHM_COMPARISON_METADATA = {
    "sa_ghmappo": {
        "family": "prediction-aware graph hierarchical controller PPO",
        "learning_type": "on-policy CTDE",
        "contract_granularity": "controller-level cache/execution/handoff",
        "uses_sa_only_mechanisms": "yes",
        "main_difference": "graph encoder, baseline/calibrated predictor features, DAG-aware diagnostics, mechanism losses, masks, and guards",
        "literature_anchor": "main method at the VEC DAG/cache/handoff intersection",
    },
    "ppo": {
        "family": "single-agent PPO",
        "learning_type": "on-policy",
        "contract_granularity": "flat five-action controller",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "flat semantic encoder and independent critic",
        "literature_anchor": "standard on-policy policy-gradient comparator",
    },
    "mappo": {
        "family": "controller-level MAPPO",
        "learning_type": "on-policy CTDE",
        "contract_granularity": "three controller heads with centralized critic",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "controller-head anti-collapse credit without graph/surrogate/guard mechanisms",
        "literature_anchor": "standard CTDE MAPPO-family comparator",
    },
    "dqn": {
        "family": "DQN",
        "learning_type": "off-policy value-based",
        "contract_granularity": "flat five-action controller",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "replay-based value learning over the shared semantic action space",
        "literature_anchor": "standard value-based deep-RL control comparator",
    },
    "ddqn": {
        "family": "Double-DQN",
        "learning_type": "off-policy value-based",
        "contract_granularity": "flat five-action controller",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "Double-DQN target selection over the same flat semantic action space",
        "literature_anchor": "optional DQN-family completeness comparator",
    },
    "dueling_dqn": {
        "family": "Dueling-DQN",
        "learning_type": "off-policy value-based",
        "contract_granularity": "flat five-action controller",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "dueling value/advantage streams over the shared semantic action space",
        "literature_anchor": "standard dueling value-based deep-RL comparator",
    },
    "dueling_ddqn": {
        "family": "Dueling Double-DQN",
        "learning_type": "off-policy value-based",
        "contract_granularity": "flat five-action controller",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "dueling value/advantage streams plus Double-DQN target selection",
        "literature_anchor": "optional DQN-family completeness comparator",
    },
    "qmix": {
        "family": "QMIX",
        "learning_type": "off-policy value decomposition",
        "contract_granularity": "controller-level cache/execution/handoff mixer",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "monotonic centralized mixer for controller Q-heads",
        "literature_anchor": "standard value-decomposition MARL comparator",
    },
    "controller_mat": {
        "family": "Multi-Agent Transformer PPO",
        "learning_type": "on-policy transformer CTDE",
        "contract_granularity": "controller tokens",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "transformer controller-token critic without graph/surrogate/guard mechanisms",
        "literature_anchor": "transformer-based multi-agent policy comparator",
    },
    "dag_offload_drl": {
        "family": "DAG offloading DRL",
        "learning_type": "on-policy PPO-style",
        "contract_granularity": "controller-level with DAG scalar block",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "DAG progress/frontier/critical-path scalars without graph message passing",
        "literature_anchor": "recent DAG/dependency-aware MEC offloading comparator",
    },
    "cache_offload_drl": {
        "family": "model-cache/offloading DRL",
        "learning_type": "on-policy PPO-style",
        "contract_granularity": "controller-level with cache scalar block",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "cache occupancy, readiness, demand, and future-load scalars",
        "literature_anchor": "recent VEC service/model-cache offloading comparator",
    },
    "dt_handoff_drl": {
        "family": "digital-twin handoff DRL",
        "learning_type": "on-policy PPO-style",
        "contract_granularity": "controller-level with DT handoff scalar block",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "raw DT sequence, dwell time, confidence, and boundary-pressure scalars",
        "literature_anchor": "recent digital-twin and mobility/handoff DRL comparator",
    },
    "ippo": {
        "family": "IPPO",
        "learning_type": "diagnostic",
        "contract_granularity": "contract-blocked under current wrapper",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "not a paper-grade independent per-agent baseline until wrapper contract is frozen",
        "literature_anchor": "diagnostic only under the current single-wrapper decision stream",
    },
    "reactive_greedy": {
        "family": "hand-written heuristic",
        "learning_type": "non-learning",
        "contract_granularity": "shared semantic action contract",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "supplementary reactive reference, not an RL comparator",
        "literature_anchor": "non-learning mechanism sanity reference",
    },
    "popularity_cache_heuristic": {
        "family": "hand-written heuristic",
        "learning_type": "non-learning",
        "contract_granularity": "shared semantic action contract",
        "uses_sa_only_mechanisms": "no",
        "main_difference": "supplementary cache-popularity reference, not an RL comparator",
        "literature_anchor": "non-learning cache-policy sanity reference",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final comparison artifacts for the paper package.")
    parser.add_argument("--final_run_root", type=Path, default=DEFAULT_FINAL_RUN_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--candidate_agent", type=str, default="sa_ghmappo")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=7)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base / path


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_value(value: Any, digits: int = 6) -> float | None:
    number = safe_float(value)
    return round(number, digits) if number is not None else None


def mean_float(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std_float(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def metric_mean(summary: dict[str, Any], agent_name: str, metric_name: str) -> float | None:
    metric = (
        summary.get("aggregate_by_agent", {})
        .get(agent_name, {})
        .get("metrics", {})
        .get(metric_name, {})
    )
    return round_value(metric.get("mean")) if isinstance(metric, dict) else None


def unique_ordered(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def agent_role(agent_name: str, learned_agents: list[str], heuristic_agents: list[str]) -> str:
    if agent_name == "sa_ghmappo":
        return "main_method"
    if agent_name in learned_agents:
        return "paper_grade_learned_baseline"
    if agent_name in heuristic_agents:
        return "supplementary_heuristic_reference"
    return "diagnostic_or_auxiliary"


def protocol_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, bool):
            if bool(actual_value) is not expected_value:
                return False
        elif isinstance(expected_value, float):
            try:
                if abs(float(actual_value) - expected_value) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    return True


def mappo_protocol_payload(final_gate: dict[str, Any]) -> dict[str, Any]:
    protocols = final_gate.get("baseline_protocol_versions", {}) or {}
    payload = protocols.get("mappo", {}) if isinstance(protocols, dict) else {}
    return dict(payload) if isinstance(payload, dict) else {}


def suite_label(report: dict[str, Any], fallback: str) -> str:
    offset = report.get("window_rank_offset")
    if offset is None:
        return fallback
    return "formal_offset_0" if int(offset) == 0 else f"holdout_offset_{offset}"


def collect_suite_reports(final_gate: dict[str, Any], final_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    reports: list[tuple[str, Path, dict[str, Any]]] = []
    formal_path = resolve_path(str(final_gate.get("formal_report_path", "")), final_root)
    if formal_path.exists():
        formal_report = load_json(formal_path)
        reports.append((suite_label(formal_report, "formal"), formal_path, formal_report))
    for index, path_text in enumerate(final_gate.get("holdout_report_paths", []) or [], start=1):
        holdout_path = resolve_path(str(path_text), final_root)
        if not holdout_path.exists():
            continue
        holdout_report = load_json(holdout_path)
        reports.append((suite_label(holdout_report, f"holdout_{index}"), holdout_path, holdout_report))
    return reports


def collect_main_metric_rows(
    suite_reports: list[tuple[str, Path, dict[str, Any]]],
    final_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_name, _, report in suite_reports:
        learned_agents = list(report.get("learned_baseline_agents", []) or [])
        heuristic_agents = list(report.get("heuristic_reference_agents", []) or [])
        agents = unique_ordered(["sa_ghmappo", *learned_agents, *heuristic_agents])
        for mode_report in report.get("mode_reports", []) or []:
            aggregate_path = resolve_path(str(mode_report.get("aggregate_summary_path", "")), final_root)
            summary = load_json(aggregate_path) if aggregate_path.exists() else {}
            for agent_name in agents:
                row: dict[str, Any] = {
                    "suite": suite_name,
                    "window_rank_offset": report.get("window_rank_offset"),
                    "mode": mode_report.get("mode"),
                    "agent_name": agent_name,
                    "role": agent_role(agent_name, learned_agents, heuristic_agents),
                    "episode_count": mode_report.get("episode_count"),
                    "aggregate_summary_path": str(aggregate_path),
                }
                for metric_name in DEFAULT_METRICS:
                    row[metric_name] = metric_mean(summary, agent_name, metric_name)
                rows.append(row)
    return rows


def benchmark_rows_path_from_mode(mode_report: dict[str, Any], final_root: Path) -> Path | None:
    aggregate_path_text = str(mode_report.get("aggregate_summary_path", ""))
    if not aggregate_path_text:
        return None
    aggregate_path = resolve_path(aggregate_path_text, final_root)
    return aggregate_path.parent / "benchmark_rows.csv"


def mean_metric_from_rows(rows: list[dict[str, Any]], metric_name: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := safe_float(row.get(metric_name))) is not None
    ]
    return round_value(mean_float(values)) if values else None


def ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    num = safe_float(numerator)
    den = safe_float(denominator)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return round_value(num / den)


def collect_action_mix_summary_rows(
    suite_reports: list[tuple[str, Path, dict[str, Any]]],
    final_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_name, _, report in suite_reports:
        learned_agents = list(report.get("learned_baseline_agents", []) or [])
        available_agents = unique_ordered(["sa_ghmappo", *learned_agents])
        audit_agents = [agent for agent in ACTION_MIX_AUDIT_AGENTS if agent in available_agents]
        for mode_report in report.get("mode_reports", []) or []:
            benchmark_path = benchmark_rows_path_from_mode(mode_report, final_root)
            if benchmark_path is None or not benchmark_path.exists():
                continue
            raw_rows = load_csv(benchmark_path)
            mode_name = str(mode_report.get("mode", ""))
            for agent_name in audit_agents:
                agent_rows = [
                    row
                    for row in raw_rows
                    if row.get("agent_name") == agent_name
                ]
                if not agent_rows:
                    continue
                output_row: dict[str, Any] = {
                    "suite": suite_name,
                    "split": suite_display(suite_name),
                    "window_rank_offset": report.get("window_rank_offset"),
                    "mode": mode_name,
                    "window_protocol": mode_display(mode_name),
                    "agent_name": agent_name,
                    "display_name": agent_label(agent_name),
                    "row_count": len(agent_rows),
                    "benchmark_rows_path": str(benchmark_path),
                }
                for metric_name in ACTION_MIX_METRICS:
                    output_row[metric_name] = mean_metric_from_rows(agent_rows, metric_name)
                execution_total = sum(
                    safe_float(output_row.get(metric_name)) or 0.0
                    for metric_name in (
                        "local_exec_count",
                        "current_rsu_exec_count",
                        "next_rsu_exec_count",
                        "neighbor_rsu_exec_count",
                        "cloud_exec_count",
                    )
                )
                mechanism_total = sum(
                    safe_float(output_row.get(metric_name)) or 0.0
                    for metric_name in ("prefetch_action_count", "migration_action_count")
                )
                output_row["execution_total_count"] = round_value(execution_total)
                output_row["mechanism_action_count"] = round_value(mechanism_total)
                output_row["local_exec_share"] = ratio_or_none(
                    output_row.get("local_exec_count"),
                    execution_total,
                )
                output_row["current_rsu_exec_share"] = ratio_or_none(
                    output_row.get("current_rsu_exec_count"),
                    execution_total,
                )
                output_row["prefetch_per_service_success"] = ratio_or_none(
                    output_row.get("prefetch_action_count"),
                    output_row.get("service_success_count"),
                )
                output_row["migration_per_service_success"] = ratio_or_none(
                    output_row.get("migration_action_count"),
                    output_row.get("service_success_count"),
                )
                output_row["prefetch_share_of_mechanism_actions"] = ratio_or_none(
                    output_row.get("prefetch_action_count"),
                    mechanism_total,
                )
                rows.append(output_row)
    return rows


def action_mix_value(row: dict[str, Any], metric_name: str) -> float | None:
    return safe_float(row.get(metric_name))


def metric_delta(row: dict[str, Any], reference_row: dict[str, Any], metric_name: str) -> float | None:
    lhs = action_mix_value(row, metric_name)
    rhs = action_mix_value(reference_row, metric_name)
    if lhs is None or rhs is None:
        return None
    return round_value(lhs - rhs)


def diagnose_mappo_action_mix(
    mappo_row: dict[str, Any],
    reference_row: dict[str, Any],
    reward_delta: float | None,
    continuity_delta: float | None,
) -> tuple[str, str]:
    signals: list[str] = []
    if (
        (safe_float(mappo_row.get("prefetch_action_count")) or 0.0)
        < 0.5 * (safe_float(reference_row.get("prefetch_action_count")) or 0.0)
    ):
        signals.append("prefetch_underuse")
    if (
        (safe_float(mappo_row.get("current_rsu_exec_count")) or 0.0)
        < 0.75 * (safe_float(reference_row.get("current_rsu_exec_count")) or 0.0)
    ):
        signals.append("current_rsu_exec_underuse")
    if (
        (safe_float(mappo_row.get("local_exec_count")) or 0.0)
        > (safe_float(reference_row.get("local_exec_count")) or 0.0) + 1.0
    ):
        signals.append("local_exec_overuse")
    if (
        (safe_float(mappo_row.get("migration_action_count")) or 0.0)
        > (safe_float(reference_row.get("migration_action_count")) or 0.0) + 1.0
    ):
        signals.append("migration_overuse")
    risk_high = bool(
        reward_delta is not None
        and reward_delta <= MAPPO_REWARD_RISK_DELTA
        and continuity_delta is not None
        and continuity_delta <= MAPPO_CONTINUITY_RISK_DELTA
        and signals
    )
    risk_level = "high" if risk_high else "tracked"
    diagnosis = ";".join(signals) if signals else "no_action_mix_collapse_signal"
    return risk_level, diagnosis


def build_mappo_action_mix_audit_rows(
    action_mix_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row.get("suite"), row.get("mode"), row.get("agent_name")): row
        for row in action_mix_rows
    }
    suite_mode_keys = unique_ordered(
        [
            f"{row.get('suite')}|{row.get('mode')}"
            for row in action_mix_rows
            if row.get("agent_name") == "mappo"
        ]
    )
    output_rows: list[dict[str, Any]] = []
    for key_text in suite_mode_keys:
        suite_name, mode_name = key_text.split("|", 1)
        mappo_row = by_key.get((suite_name, mode_name, "mappo"))
        if not mappo_row:
            continue
        for reference_agent in MAPPO_RISK_REFERENCE_AGENTS:
            reference_row = by_key.get((suite_name, mode_name, reference_agent))
            if not reference_row:
                continue
            reward_delta = metric_delta(mappo_row, reference_row, "total_reward")
            continuity_delta = metric_delta(mappo_row, reference_row, "workflow_continuity_rate")
            failure_delta = metric_delta(mappo_row, reference_row, "handoff_failure_rate")
            risk_level, diagnosis = diagnose_mappo_action_mix(
                mappo_row,
                reference_row,
                reward_delta,
                continuity_delta,
            )
            output_rows.append(
                {
                    "suite": suite_name,
                    "split": suite_display(suite_name),
                    "mode": mode_name,
                    "window_protocol": mode_display(mode_name),
                    "reference_agent": reference_agent,
                    "reference": agent_label(reference_agent),
                    "mappo_minus_reference_reward": reward_delta,
                    "mappo_minus_reference_continuity": continuity_delta,
                    "mappo_minus_reference_handoff_failure": failure_delta,
                    "mappo_reward": round_value(mappo_row.get("total_reward")),
                    "reference_reward": round_value(reference_row.get("total_reward")),
                    "mappo_continuity": round_value(mappo_row.get("workflow_continuity_rate")),
                    "reference_continuity": round_value(reference_row.get("workflow_continuity_rate")),
                    "mappo_handoff_failure": round_value(mappo_row.get("handoff_failure_rate")),
                    "reference_handoff_failure": round_value(reference_row.get("handoff_failure_rate")),
                    "mappo_service_success_count": round_value(mappo_row.get("service_success_count")),
                    "reference_service_success_count": round_value(reference_row.get("service_success_count")),
                    "mappo_prefetch_action_count": round_value(mappo_row.get("prefetch_action_count")),
                    "reference_prefetch_action_count": round_value(reference_row.get("prefetch_action_count")),
                    "mappo_migration_action_count": round_value(mappo_row.get("migration_action_count")),
                    "reference_migration_action_count": round_value(reference_row.get("migration_action_count")),
                    "mappo_local_exec_count": round_value(mappo_row.get("local_exec_count")),
                    "reference_local_exec_count": round_value(reference_row.get("local_exec_count")),
                    "mappo_current_rsu_exec_count": round_value(mappo_row.get("current_rsu_exec_count")),
                    "reference_current_rsu_exec_count": round_value(reference_row.get("current_rsu_exec_count")),
                    "risk_level": risk_level,
                    "diagnosis": diagnosis,
                    "paper_action": (
                        "Report as a controller-level MAPPO action-mix limitation; "
                        "state strongest learned-baseline claims against PPO when PPO is strongest."
                    ),
                }
            )
    return output_rows


def collect_margin_rows(
    suite_reports: list[tuple[str, Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_name, _, report in suite_reports:
        learned_agents = list(report.get("learned_baseline_agents", []) or [])
        heuristic_agents = list(report.get("heuristic_reference_agents", []) or [])
        for mode_report in report.get("mode_reports", []) or []:
            rewards = (mode_report.get("metrics", {}) or {}).get("total_reward", {}) or {}
            strongest_heuristic = ""
            strongest_heuristic_reward: float | None = None
            for heuristic_agent in heuristic_agents:
                reward = safe_float(rewards.get(heuristic_agent))
                if reward is not None and (
                    strongest_heuristic_reward is None or reward > strongest_heuristic_reward
                ):
                    strongest_heuristic = heuristic_agent
                    strongest_heuristic_reward = reward
            sa_reward = safe_float(rewards.get("sa_ghmappo"))
            learned_reward = safe_float(mode_report.get("strongest_learned_baseline_reward"))
            row = {
                "suite": suite_name,
                "window_rank_offset": report.get("window_rank_offset"),
                "mode": mode_report.get("mode"),
                "passed": bool(mode_report.get("passed")),
                "episode_count": mode_report.get("episode_count"),
                "learned_baseline_set": " ".join(learned_agents),
                "strongest_learned_baseline": mode_report.get("strongest_learned_baseline", ""),
                "sa_total_reward": round_value(sa_reward),
                "strongest_learned_baseline_reward": round_value(learned_reward),
                "sa_minus_strongest_learned_reward": round_value(
                    None if sa_reward is None or learned_reward is None else sa_reward - learned_reward
                ),
                "strongest_heuristic_reference": strongest_heuristic,
                "strongest_heuristic_reference_reward": round_value(strongest_heuristic_reward),
                "sa_minus_strongest_heuristic_reward": round_value(
                    None
                    if sa_reward is None or strongest_heuristic_reward is None
                    else sa_reward - strongest_heuristic_reward
                ),
                "claim_role": "primary_learned_gate",
                "heuristic_policy": "supplementary_reference_only",
            }
            rows.append(row)
    return rows


def load_existing_statistics(
    suite_reports: list[tuple[str, Path, dict[str, Any]]],
    final_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_name, _, report in suite_reports:
        statistics_path_text = str(report.get("statistics_path", ""))
        if not statistics_path_text:
            continue
        statistics_path = resolve_path(statistics_path_text, final_root)
        if not statistics_path.exists():
            continue
        for row in load_csv(statistics_path):
            output_row: dict[str, Any] = {"suite": suite_name, "statistics_path": str(statistics_path)}
            output_row.update(row)
            ci_low = safe_float(row.get("ci95_low"))
            output_row["claim_evidence"] = "positive_ci" if ci_low is not None and ci_low > 0 else "weak_or_mixed_ci"
            rows.append(output_row)
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_sign_test_pvalue(wins: int, losses: int) -> float:
    try:
        wins = int(wins)
        losses = int(losses)
    except (TypeError, ValueError):
        return 1.0
    trials = wins + losses
    if trials <= 0:
        return 1.0
    tail = min(wins, losses)
    cumulative = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * cumulative)


def summarize_deltas(
    deltas: list[float],
    clusters: list[tuple[str, ...]],
    bootstrap_samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    if not deltas:
        return {
            "paired_count": 0,
            "cluster_count": 0,
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "std_delta": 0.0,
            "cohen_dz": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "sign_test_pvalue": 1.0,
        }
    by_cluster: dict[tuple[str, ...], list[float]] = {}
    for delta, cluster in zip(deltas, clusters):
        by_cluster.setdefault(cluster, []).append(delta)
    cluster_summaries = [(sum(values), len(values)) for values in by_cluster.values()]
    del rng
    state = (len(deltas) * 1_000_003 + len(cluster_summaries) * 9_176 + bootstrap_samples * 97 + 31) & 0x7FFFFFFF
    bootstrap_means: list[float] = []
    sample_iterations = max(1, int(bootstrap_samples))
    cluster_total = len(cluster_summaries)
    sample_index = 0
    while sample_index < sample_iterations:
        sample_sum = 0.0
        sample_count = 0
        cluster_index = 0
        while cluster_index < cluster_total:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            cluster_sum, cluster_count = cluster_summaries[state % len(cluster_summaries)]
            sample_sum += cluster_sum
            sample_count += cluster_count
            cluster_index += 1
        bootstrap_means.append(sample_sum / sample_count if sample_count else 0.0)
        sample_index += 1
    mean_delta = mean_float(deltas)
    std_delta = sample_std_float(deltas, mean_delta)
    win_count = sum(1 for value in deltas if value > 1e-9)
    loss_count = sum(1 for value in deltas if value < -1e-9)
    tie_count = len(deltas) - win_count - loss_count
    return {
        "paired_count": len(deltas),
        "cluster_count": len(by_cluster),
        "mean_delta": round(mean_delta, 6),
        "ci95_low": round(percentile(bootstrap_means, 0.025), 6),
        "ci95_high": round(percentile(bootstrap_means, 0.975), 6),
        "std_delta": round(std_delta, 6),
        "cohen_dz": round(mean_delta / std_delta, 6) if std_delta > 1e-12 else 0.0,
        "wins": win_count,
        "ties": tie_count,
        "losses": loss_count,
        "sign_test_pvalue": round(exact_sign_test_pvalue(win_count, loss_count), 6),
    }


def row_key(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(f"{field}={row.get(field, '')}" for field in fields if row.get(field, "") != "")


def compute_paired_statistics(
    rows: list[dict[str, str]],
    *,
    suite_name: str,
    support_kind: str,
    setting_id: str,
    setting_field: str,
    candidate_agent: str,
    baseline_agents: list[str],
    bootstrap_samples: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pair_fields = ["seed", "window_id", "workflow_id", setting_field]
    by_key: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = row_key(row, pair_fields)
        by_key.setdefault(key, {})[row.get("agent_name", "")] = row

    output_rows: list[dict[str, Any]] = []
    for baseline_agent in baseline_agents:
        for metric_name in DEFAULT_METRICS:
            signed_deltas: list[float] = []
            raw_deltas: list[float] = []
            clusters: list[tuple[str, ...]] = []
            for key, agents_by_key in by_key.items():
                candidate_row = agents_by_key.get(candidate_agent)
                baseline_row = agents_by_key.get(baseline_agent)
                if candidate_row is None or baseline_row is None:
                    continue
                candidate_value = safe_float(candidate_row.get(metric_name))
                baseline_value = safe_float(baseline_row.get(metric_name))
                if candidate_value is None or baseline_value is None:
                    continue
                raw_delta = candidate_value - baseline_value
                raw_deltas.append(raw_delta)
                signed_deltas.append(-raw_delta if metric_name in LOWER_IS_BETTER else raw_delta)
                clusters.append(key)
            summary = summarize_deltas(signed_deltas, clusters, bootstrap_samples, rng)
            raw_summary = summarize_deltas(raw_deltas, clusters, bootstrap_samples, rng)
            output_rows.append(
                {
                    "suite": suite_name,
                    "support_kind": support_kind,
                    "setting_id": setting_id,
                    "candidate_agent": candidate_agent,
                    "baseline_agent": baseline_agent,
                    "metric": metric_name,
                    "higher_is_better": metric_name not in LOWER_IS_BETTER,
                    "signed_positive_favors_candidate": True,
                    **summary,
                    "raw_mean_delta_candidate_minus_baseline": raw_summary["mean_delta"],
                    "raw_ci95_low": raw_summary["ci95_low"],
                    "raw_ci95_high": raw_summary["ci95_high"],
                    "claim_evidence": "positive_ci" if summary["ci95_low"] > 0 else "weak_or_mixed_ci",
                }
            )
    return output_rows


def collect_support_rows(
    final_gate: dict[str, Any],
    final_root: Path,
    *,
    candidate_agent: str,
    bootstrap_samples: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    support_agents = list(final_gate.get("support_agents", []) or [])
    learned_baselines = [agent for agent in support_agents if agent != candidate_agent]
    for support_kind, summary_path_text in (final_gate.get("support_paths", {}) or {}).items():
        if support_kind not in SUPPORT_ROW_FILES:
            continue
        summary_path = resolve_path(str(summary_path_text), final_root)
        row_file_name, setting_field = SUPPORT_ROW_FILES[support_kind]
        rows_path = summary_path.parent / row_file_name
        if not rows_path.exists():
            continue
        raw_rows = load_csv(rows_path)
        grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
        for row in raw_rows:
            agent_name = row.get("agent_name", "")
            if agent_name not in support_agents:
                continue
            setting = row.get(setting_field) or row.get("setting_id") or "all"
            key = (agent_name, setting)
            grouped.setdefault(key, {metric: [] for metric in DEFAULT_METRICS})
            for metric_name in DEFAULT_METRICS:
                value = safe_float(row.get(metric_name))
                if value is not None:
                    grouped[key][metric_name].append(value)
        for (agent_name, setting), metrics in sorted(grouped.items()):
            output_row: dict[str, Any] = {
                "support_kind": support_kind,
                "setting_id": setting,
                "agent_name": agent_name,
                "role": agent_role(agent_name, learned_baselines, []),
                "rows_path": str(rows_path),
            }
            for metric_name, values in metrics.items():
                output_row[metric_name] = round(mean_float(values), 6) if values else None
            summary_rows.append(output_row)
        stat_rows.extend(
            compute_paired_statistics(
                [row for row in raw_rows if row.get("agent_name", "") in support_agents],
                suite_name="support",
                support_kind=support_kind,
                setting_id="all",
                setting_field=setting_field,
                candidate_agent=candidate_agent,
                baseline_agents=learned_baselines,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
        )
        setting_ids = sorted(
            {
                row.get(setting_field) or row.get("setting_id") or "all"
                for row in raw_rows
                if row.get("agent_name", "") in support_agents
            }
        )
        for setting_id in setting_ids:
            setting_rows = [
                row
                for row in raw_rows
                if row.get("agent_name", "") in support_agents
                and (row.get(setting_field) or row.get("setting_id") or "all") == setting_id
            ]
            stat_rows.extend(
                compute_paired_statistics(
                    setting_rows,
                    suite_name="support",
                    support_kind=support_kind,
                    setting_id=setting_id,
                    setting_field=setting_field,
                    candidate_agent=candidate_agent,
                    baseline_agents=learned_baselines,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                )
            )
    return summary_rows, stat_rows


def build_protocol_rows(final_gate: dict[str, Any]) -> list[dict[str, Any]]:
    learned_agents = list(final_gate.get("learned_baseline_agents", []) or DEFAULT_LEARNED_BASELINES)
    heuristic_agents = list(final_gate.get("heuristic_reference_agents", []) or DEFAULT_HEURISTIC_REFERENCES)
    mappo_protocol = mappo_protocol_payload(final_gate)
    mappo_protocol_current = protocol_matches(CURRENT_MAPPO_PROTOCOL, mappo_protocol)
    rows = [
        {
            "agent_name": "sa_ghmappo",
            "role": "main_method",
            "paper_grade_role": "candidate",
            "included_in_primary_gate": True,
            "trainable_now": True,
            "contract_status": "valid",
            "comparison_note": "Surrogate-assisted graph hierarchical PPO policy under the handoff-pressure VEC workflow contract.",
        }
    ]
    for agent_name in learned_agents:
        contract_status = "valid"
        protocol_version = ""
        comparison_note = "Clean retrained by the final-submission suite and audited by duplicate-trace and provenance gates."
        if agent_name == "mappo":
            protocol_version = (
                "aggregation_reason_weighted_controller_ppo_v3"
                if mappo_protocol_current
                else "missing_or_pre_v3_head_credit"
            )
            contract_status = "valid" if mappo_protocol_current else "pre_head_credit_protocol_missing"
            comparison_note = (
                "Controller-level CTDE MAPPO with aggregation-reason controller head-credit v3. "
                "Pre-v3/pre-head-credit MAPPO checkpoints are archived only and cannot support a current MAPPO claim."
            )
        rows.append(
            {
                "agent_name": agent_name,
                "role": "paper_grade_learned_baseline",
                "paper_grade_role": "primary_learned_comparator",
                "included_in_primary_gate": True,
                "trainable_now": True,
                "contract_status": contract_status,
                "protocol_version": protocol_version,
                "comparison_note": comparison_note,
            }
        )
    if "ippo" not in learned_agents:
        rows.append(
            {
                "agent_name": "ippo",
                "role": "diagnostic_baseline",
                "paper_grade_role": "excluded",
                "included_in_primary_gate": False,
                "trainable_now": False,
                "contract_status": "contract_blocked",
                "comparison_note": "Current single-wrapper stream cannot support an independent IPPO claim.",
            }
        )
    if "mappo" not in learned_agents:
        rows.append(
            {
                "agent_name": "mappo",
                "role": "available_learned_baseline",
                "paper_grade_role": "not_in_current_primary_gate",
                "included_in_primary_gate": False,
                "trainable_now": True,
                "contract_status": "valid_controller_level_ctde_head_credit",
                "protocol_version": "aggregation_reason_weighted_controller_ppo_v3",
                "comparison_note": "Controller-level MAPPO is available for new runs with aggregation-reason controller head-credit v3; this artifact does not include it unless listed in learned_baseline_agents.",
            }
        )
    if "qmix" not in learned_agents:
        rows.append(
            {
                "agent_name": "qmix",
                "role": "available_learned_baseline",
                "paper_grade_role": "not_in_current_primary_gate",
                "included_in_primary_gate": False,
                "trainable_now": True,
                "contract_status": "valid_controller_level_value_decomposition",
                "comparison_note": "Controller-level QMIX is available for new runs; this artifact does not include it unless listed in learned_baseline_agents.",
            }
        )
    if "controller_mat" not in learned_agents:
        rows.append(
            {
                "agent_name": "controller_mat",
                "role": "available_learned_baseline",
                "paper_grade_role": "not_in_current_primary_gate",
                "included_in_primary_gate": False,
                "trainable_now": True,
                "contract_status": "valid_controller_level_transformer_ctde",
                "comparison_note": "Controller-level MAT-style transformer baseline is available for new runs; this artifact does not include it unless listed in learned_baseline_agents.",
            }
        )
    domain_available = {
        "dag_offload_drl": (
            "valid_domain_dag_offload",
            "Dependency-aware DAG offloading DRL baseline is available for new runs; this artifact does not include it unless listed in learned_baseline_agents.",
        ),
        "cache_offload_drl": (
            "valid_domain_model_cache_offload",
            "Model/adapter cache-aware offloading DRL baseline is available for new runs; this artifact does not include it unless listed in learned_baseline_agents.",
        ),
        "dt_handoff_drl": (
            "valid_domain_digital_twin_handoff",
            "Digital-twin handoff/service-migration DRL baseline is available for new runs; this artifact does not include it unless listed in learned_baseline_agents.",
        ),
    }
    for agent_name, (contract_status, comparison_note) in domain_available.items():
        if agent_name not in learned_agents:
            rows.append(
                {
                    "agent_name": agent_name,
                    "role": "available_domain_learned_baseline",
                    "paper_grade_role": "not_in_current_primary_gate",
                    "included_in_primary_gate": False,
                    "trainable_now": True,
                    "contract_status": contract_status,
                    "comparison_note": comparison_note,
                }
            )
    optional_dqn_variants = {
        "ddqn": "May be added only if its benchmark traces are independent from DQN.",
        "dueling_ddqn": "May be added only if its benchmark traces are independent from Dueling-DQN.",
    }
    for agent_name, comparison_note in optional_dqn_variants.items():
        if agent_name not in learned_agents:
            rows.append(
                {
                    "agent_name": agent_name,
                    "role": "optional_dqn_family_variant",
                    "paper_grade_role": "not_in_current_primary_gate",
                    "included_in_primary_gate": False,
                    "trainable_now": True,
                    "contract_status": "optional_after_duplicate_trace_audit",
                    "comparison_note": comparison_note,
                }
            )
    for agent_name in heuristic_agents:
        rows.append(
            {
                "agent_name": agent_name,
                "role": "supplementary_heuristic_reference",
                "paper_grade_role": "supplementary_reference",
                "included_in_primary_gate": False,
                "trainable_now": False,
                "contract_status": "valid_non_learning_reference",
                "comparison_note": "Reported for mechanism sanity checks, not as a learned-baseline gate blocker.",
            }
        )
    return rows


def find_stat(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    baseline_agent: str,
    metric: str,
) -> dict[str, Any] | None:
    for row in rows:
        if row.get("suite") == suite and row.get("baseline_agent") == baseline_agent and row.get("metric") == metric:
            return row
    return None


def build_claim_checks(
    final_gate: dict[str, Any],
    suite_reports: list[tuple[str, Path, dict[str, Any]]],
    margin_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    support_stat_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks = [
        {
            "check": "final_gate_paper_claim_ready",
            "passed": bool(final_gate.get("paper_claim_ready")),
            "detail": f"blockers={final_gate.get('blockers', [])}",
        },
        {
            "check": "formal_training_provenance",
            "passed": bool(final_gate.get("formal_training_provenance", {}).get("passed")),
            "detail": f"record_count={final_gate.get('formal_training_provenance', {}).get('record_count')}",
        },
    ]
    for suite_name, _, report in suite_reports:
        checks.append(
            {
                "check": f"{suite_name}_contract_and_independence",
                "passed": bool(
                    report.get("baseline_contract", {}).get("passed")
                    and report.get("baseline_independence", {}).get("passed")
                ),
                "detail": (
                    f"contract_blockers={report.get('baseline_contract', {}).get('blockers', [])}; "
                    f"independence_blockers={report.get('baseline_independence', {}).get('blockers', [])}"
                ),
            }
        )
        learned_agents = list(report.get("learned_baseline_agents", []) or [])
        reward_stats = [
            find_stat(paired_rows, suite=suite_name, baseline_agent=agent_name, metric="total_reward")
            for agent_name in learned_agents
        ]
        missing_agents = [
            agent_name
            for agent_name, stat in zip(learned_agents, reward_stats)
            if not stat
        ]
        weak_stats = [
            stat
            for stat in reward_stats
            if stat
            and safe_float(stat.get("ci95_low")) is not None
            and float(stat["ci95_low"]) <= 0.0
        ]
        checks.append(
            {
                "check": f"{suite_name}_reward_ci_vs_all_learned",
                "passed": bool(learned_agents)
                and not missing_agents
                and not weak_stats
                and all(
                    stat
                    and safe_float(stat.get("ci95_low")) is not None
                    and float(stat["ci95_low"]) > 0.0
                    for stat in reward_stats
                ),
                "detail": (
                    f"missing={missing_agents}; weak="
                    f"{[(stat.get('baseline_agent'), stat.get('ci95_low')) for stat in weak_stats]}; "
                    + "; ".join(
                        f"{stat.get('baseline_agent')} mean={stat.get('mean_delta')} "
                        f"ci=[{stat.get('ci95_low')}, {stat.get('ci95_high')}]"
                        for stat in reward_stats
                        if stat
                    )
                ),
            }
        )
    positive_margins = [
        row for row in margin_rows if safe_float(row.get("sa_minus_strongest_learned_reward")) is not None
    ]
    checks.append(
        {
            "check": "all_modes_positive_margin_vs_strongest_learned",
            "passed": bool(positive_margins)
            and all(float(row["sa_minus_strongest_learned_reward"]) > 0.0 for row in positive_margins),
            "detail": "; ".join(
                f"{row.get('suite')}:{row.get('mode')}={row.get('sa_minus_strongest_learned_reward')}"
                for row in positive_margins
            ),
        }
    )
    required_prediction_settings = list(final_gate.get("prediction_required_settings", []) or [])
    for setting_id in required_prediction_settings:
        ppo_rows = [
            row
            for row in support_stat_rows
            if row.get("support_kind") == "prediction"
            and row.get("setting_id") == setting_id
            and row.get("baseline_agent") == "ppo"
            and row.get("metric") == "total_reward"
        ]
        ppo_row = ppo_rows[0] if ppo_rows else None
        checks.append(
            {
                "check": f"prediction_setting_ci_vs_ppo:{setting_id}",
                "passed": bool(
                    ppo_row
                    and safe_float(ppo_row.get("ci95_low")) is not None
                    and float(ppo_row["ci95_low"]) > 0.0
                ),
                "detail": (
                    "missing"
                    if not ppo_row
                    else f"mean_delta={ppo_row.get('mean_delta')}, ci95=[{ppo_row.get('ci95_low')}, {ppo_row.get('ci95_high')}]"
                ),
            }
        )
    return checks


def agent_label(agent_name: str) -> str:
    return AGENT_LABELS.get(agent_name, agent_name)


def suite_display(suite_name: str) -> str:
    return SUITE_LABELS.get(suite_name, suite_name)


def mode_display(mode_name: str) -> str:
    return MODE_LABELS.get(mode_name, mode_name)


def support_display(support_kind: str) -> str:
    return SUPPORT_LABELS.get(support_kind, support_kind)


def percent_delta(candidate_value: Any, baseline_value: Any) -> float | None:
    candidate = safe_float(candidate_value)
    baseline = safe_float(baseline_value)
    if candidate is None or baseline is None or abs(baseline) < 1e-12:
        return None
    return round((candidate - baseline) / abs(baseline) * 100.0, 3)


def signed_text(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:+.{digits}f}"


def percent_text(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:+.{digits}f}%"


def ci_text(row: dict[str, Any], digits: int = 3) -> str:
    low = format_num(row.get("ci95_low"), digits)
    high = format_num(row.get("ci95_high"), digits)
    return f"[{low}, {high}]"


def wins_ties_losses(row: dict[str, Any]) -> str:
    return f"{row.get('wins')}/{row.get('ties')}/{row.get('losses')}"


def build_paper_main_comparison_rows(
    main_metric_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    table_agents: list[str],
) -> list[dict[str, Any]]:
    by_key = {
        (row.get("suite"), row.get("mode"), row.get("agent_name")): row
        for row in main_metric_rows
    }
    output_rows: list[dict[str, Any]] = []
    for margin in margin_rows:
        suite_name = str(margin.get("suite", ""))
        mode_name = str(margin.get("mode", ""))
        row: dict[str, Any] = {
            "split": suite_display(suite_name),
            "suite": suite_name,
            "window_protocol": mode_display(mode_name),
            "mode": mode_name,
            "episode_count": margin.get("episode_count"),
            "strongest_learned_baseline": agent_label(str(margin.get("strongest_learned_baseline", ""))),
            "strongest_learned_baseline_agent": str(margin.get("strongest_learned_baseline", "")),
            "delta_vs_strongest_learned": round_value(margin.get("sa_minus_strongest_learned_reward")),
            "strongest_heuristic_reference": agent_label(str(margin.get("strongest_heuristic_reference", ""))),
            "delta_vs_strongest_heuristic_reference": round_value(
                margin.get("sa_minus_strongest_heuristic_reward")
            ),
            "claim_scope": "primary learned-baseline comparison",
        }
        for agent_name in table_agents:
            agent_row = by_key.get((suite_name, mode_name, agent_name), {})
            row[f"{agent_name}_reward"] = round_value(agent_row.get("total_reward"))
            row[f"{agent_name}_continuity"] = round_value(agent_row.get("workflow_continuity_rate"))
            row[f"{agent_name}_handoff_failure"] = round_value(agent_row.get("handoff_failure_rate"))
        row["delta_vs_ppo"] = round_value(
            None
            if row.get("sa_ghmappo_reward") is None or row.get("ppo_reward") is None
            else float(row["sa_ghmappo_reward"]) - float(row["ppo_reward"])
        )
        row["relative_delta_vs_ppo_percent"] = percent_delta(row.get("sa_ghmappo_reward"), row.get("ppo_reward"))
        output_rows.append(row)
    return output_rows


def paper_table_role_for(status: str) -> str:
    if status == "main_method":
        return "main_method"
    if status == "primary_claimable":
        return "primary_learned_comparator"
    if status == "supplementary_only":
        return "supplementary_reference"
    return "excluded"


def audit_decision_for(status: str, contract_status: str) -> str:
    if status == "main_method":
        return "pass_main_method"
    if status == "primary_claimable" and contract_status == "valid":
        return "pass_primary_learned"
    if status == "supplementary_only" and contract_status == "valid_non_learning_reference":
        return "pass_supplementary_reference"
    return "exclude_from_main_table"


def compact_ci(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    mean_delta = signed_text(row.get("mean_delta"))
    low = format_num(row.get("ci95_low"))
    high = format_num(row.get("ci95_high"))
    return f"{mean_delta} [{low}, {high}]"


def build_audited_main_table_rows(
    main_metric_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_by_key = {
        (str(row.get("suite", "")), str(row.get("mode", "")), str(row.get("agent_name", ""))): row
        for row in main_metric_rows
    }
    protocol_order = {str(row.get("agent_name", "")): index for index, row in enumerate(protocol_rows)}
    output_rows: list[dict[str, Any]] = []
    for protocol_row in protocol_rows:
        agent_name = str(protocol_row.get("agent_name", ""))
        status = current_artifact_status(protocol_row)
        contract_status = str(protocol_row.get("contract_status", ""))
        audit_decision = audit_decision_for(status, contract_status)
        if audit_decision == "exclude_from_main_table":
            continue

        row: dict[str, Any] = {
            "agent_name": agent_name,
            "display_name": agent_label(agent_name),
            "paper_table_role": paper_table_role_for(status),
            "audit_decision": audit_decision,
            "contract_status": contract_status,
            "training_budget_policy": training_budget_policy_for(protocol_row, status),
            "claim_boundary": (
                "primary claim comparator"
                if status == "primary_claimable"
                else "supplementary reference only"
                if status == "supplementary_only"
                else "candidate method"
            ),
        }
        for suite_name, suite_prefix in [
            ("formal_offset_0", "formal"),
            ("holdout_offset_3", "holdout"),
        ]:
            stat = find_stat(paired_rows, suite=suite_name, baseline_agent=agent_name, metric="total_reward")
            row[f"{suite_prefix}_sa_minus_agent_reward_ci"] = compact_ci(stat)
            for mode_name, mode_prefix in [
                ("mixed_informative", "mixed"),
                ("full_stratified", "full"),
            ]:
                metric_row = metric_by_key.get((suite_name, mode_name, agent_name), {})
                prefix = f"{suite_prefix}_{mode_prefix}"
                row[f"{prefix}_reward"] = round_value(metric_row.get("total_reward"))
                row[f"{prefix}_continuity"] = round_value(metric_row.get("workflow_continuity_rate"))
                row[f"{prefix}_handoff_failure"] = round_value(metric_row.get("handoff_failure_rate"))
        row["audit_basis"] = str(protocol_row.get("comparison_note", ""))
        output_rows.append(row)
    return sorted(output_rows, key=lambda row: protocol_order.get(str(row.get("agent_name", "")), 10_000))


def compact_set(values: list[Any]) -> str:
    return " ".join(sorted({str(value) for value in values if value not in {None, ""}}))


def collect_training_fairness_rows(
    suite_reports: list[tuple[str, Path, dict[str, Any]]],
    final_root: Path,
) -> list[dict[str, Any]]:
    formal_report = next(
        (report for suite_name, _, report in suite_reports if suite_name == "formal_offset_0"),
        {},
    )
    training_records = list(formal_report.get("training_records", []) or [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in training_records:
        grouped.setdefault(str(record.get("agent_name", "")), []).append(record)

    required_seeds = {int(seed) for seed in formal_report.get("seeds", []) or []}
    rows: list[dict[str, Any]] = []
    for agent_name in sorted(grouped):
        summaries: list[dict[str, Any]] = []
        trained_by_suite_flags: list[bool] = []
        for record in grouped[agent_name]:
            trained_by_suite_flags.append(bool(record.get("trained_by_suite")))
            summary_path_text = str(record.get("train_summary_path", ""))
            summary_path = resolve_path(summary_path_text, final_root)
            if summary_path.exists():
                summaries.append(load_json(summary_path))
        seeds = {
            int(record.get("seed"))
            for record in grouped[agent_name]
            if str(record.get("seed", "")).lstrip("-").isdigit()
        }
        episodes = [summary.get("episodes") for summary in summaries]
        update_every = [summary.get("update_every") for summary in summaries]
        update_count = [summary.get("update_count") for summary in summaries]
        profiles = [summary.get("profile") or summary.get("config_profile") for summary in summaries]
        observation_contracts = [
            (summary.get("algo_spec", {}) or {}).get("observation_contract")
            for summary in summaries
        ]
        action_contracts = [
            (summary.get("algo_spec", {}) or {}).get("action_contract")
            for summary in summaries
        ]
        protocols = [summary.get("agent_protocol", {}) or {} for summary in summaries]
        sa_only_flags = [
            bool(protocol.get("graph_encoder"))
            or bool(protocol.get("surrogate_enhanced_head"))
            or bool(protocol.get("uses_sa_only_mechanisms"))
            or bool(protocol.get("mechanism_auxiliary_loss"))
            or bool(protocol.get("heuristic_imitation"))
            or bool(protocol.get("continuity_guard"))
            or bool(protocol.get("backhaul_guard"))
            or bool(protocol.get("cache_warm_start_guard"))
            for protocol in protocols
        ]
        budget_pass = (
            seeds == required_seeds
            and len(summaries) == len(required_seeds)
            and compact_set(episodes) == "96"
            and compact_set(update_every) == "6"
            and compact_set(update_count) == "16"
            and compact_set(profiles) == "baseline_safe"
            and all(trained_by_suite_flags)
        )
        mechanism_pass = not any(sa_only_flags)
        rows.append(
            {
                "agent_name": agent_name,
                "display_name": agent_label(agent_name),
                "audit_status": "pass" if budget_pass and mechanism_pass else "review",
                "records": len(summaries),
                "seeds": compact_set(sorted(seeds)),
                "episodes": compact_set(episodes),
                "update_every": compact_set(update_every),
                "update_count": compact_set(update_count),
                "profile": compact_set(profiles),
                "observation_contract": compact_set(observation_contracts),
                "action_contract": compact_set(action_contracts),
                "trained_by_final_suite": all(trained_by_suite_flags),
                "uses_sa_only_mechanisms": any(sa_only_flags),
                "fairness_basis": (
                    "matched environment-interaction budget; algorithm-specific optimizers preserved"
                    if budget_pass
                    else "budget or provenance mismatch requires review"
                ),
            }
        )
    return rows


def build_strongest_comparator_audit_rows(
    main_metric_rows: list[dict[str, Any]],
    learned_agents: list[str],
    heuristic_agents: list[str],
) -> list[dict[str, Any]]:
    suites = unique_ordered([str(row.get("suite", "")) for row in main_metric_rows])
    modes = unique_ordered([str(row.get("mode", "")) for row in main_metric_rows])
    rows: list[dict[str, Any]] = []
    for suite_name in suites:
        for mode_name in modes:
            split_rows = [
                row
                for row in main_metric_rows
                if row.get("suite") == suite_name and row.get("mode") == mode_name
            ]
            learned_ranked = sorted(
                [
                    row
                    for row in split_rows
                    if row.get("agent_name") in learned_agents
                    and safe_float(row.get("total_reward")) is not None
                ],
                key=lambda row: float(row["total_reward"]),
                reverse=True,
            )
            heuristic_ranked = sorted(
                [
                    row
                    for row in split_rows
                    if row.get("agent_name") in heuristic_agents
                    and safe_float(row.get("total_reward")) is not None
                ],
                key=lambda row: float(row["total_reward"]),
                reverse=True,
            )
            if not learned_ranked:
                continue
            ppo_rank = next(
                (
                    index + 1
                    for index, row in enumerate(learned_ranked)
                    if row.get("agent_name") == "ppo"
                ),
                None,
            )
            top_learned = learned_ranked[0]
            rows.append(
                {
                    "suite": suite_name,
                    "split": suite_display(suite_name),
                    "mode": mode_name,
                    "window_protocol": mode_display(mode_name),
                    "strongest_learned_agent": top_learned.get("agent_name"),
                    "strongest_learned": agent_label(str(top_learned.get("agent_name", ""))),
                    "strongest_learned_reward": round_value(top_learned.get("total_reward")),
                    "ppo_rank_among_learned": ppo_rank,
                    "ppo_is_strongest_learned": top_learned.get("agent_name") == "ppo",
                    "top3_learned_by_reward": "; ".join(
                        f"{agent_label(str(row.get('agent_name', '')))}={format_num(row.get('total_reward'))}"
                        for row in learned_ranked[:3]
                    ),
                    "strongest_heuristic": (
                        agent_label(str(heuristic_ranked[0].get("agent_name", "")))
                        if heuristic_ranked
                        else ""
                    ),
                    "strongest_heuristic_reward": (
                        round_value(heuristic_ranked[0].get("total_reward"))
                        if heuristic_ranked
                        else None
                    ),
                    "decision_rule": "rank all audited learned baselines by reward per split; do not hard-code PPO",
                }
            )
    return rows


def build_reviewer_issue_resolution_rows(
    final_gate: dict[str, Any],
    fairness_rows: list[dict[str, Any]],
    strongest_rows: list[dict[str, Any]],
    self_review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fairness_pass = all(row.get("audit_status") == "pass" for row in fairness_rows)
    ppo_all_strongest = all(row.get("ppo_is_strongest_learned") in {True, "True"} for row in strongest_rows)
    strongest_summary = "; ".join(
        f"{row.get('split')}/{row.get('window_protocol')}={row.get('strongest_learned')}"
        for row in strongest_rows
    )
    limitations = {
        str(row.get("category", "")): row
        for row in self_review_rows
        if row.get("status") == "limitation"
    }
    return [
        {
            "issue_id": "R1",
            "reviewer_concern": "Strongest comparator may not always be PPO.",
            "status": "resolved_for_current_artifact" if ppo_all_strongest else "resolved_by_dynamic_ranking",
            "evidence": strongest_summary,
            "paper_resolution": "Use the strongest audited learned baseline per split; current artifact empirically ranks PPO first in all four main splits.",
            "remaining_action": "Repeat this automatic ranking after every rerun instead of hard-coding PPO.",
        },
        {
            "issue_id": "R2",
            "reviewer_concern": "Each learned algorithm needs a fair training and evaluation budget.",
            "status": "resolved" if fairness_pass else "needs_review",
            "evidence": (
                f"{len(fairness_rows)} learned comparators audited; expected seeds="
                f"{' '.join(str(seed) for seed in final_gate.get('budget_protocol', {}).get('seeds', []))}; "
                f"episodes={final_gate.get('budget_protocol', {}).get('baseline_episodes')}; "
                f"max_steps={final_gate.get('budget_protocol', {}).get('max_steps')}"
            ),
            "paper_resolution": "State matched environment-interaction budget, shared NGSIM+Alibaba windows/workflows/seeds, and preserved algorithm-specific optimizer internals.",
            "remaining_action": "For stronger rebuttal, add a hyperparameter appendix listing per-algorithm optimizer settings.",
        },
        {
            "issue_id": "R3",
            "reviewer_concern": "MAPPO is much weaker than PPO; this may indicate an unfair baseline.",
            "status": "limitation_with_diagnostic",
            "evidence": limitations.get("MAPPO action-mix risk", {}).get("evidence", "MAPPO action-mix audit generated."),
            "paper_resolution": "Keep MAPPO in the table as controller-level CTDE with head-credit, report action-mix collapse, and avoid relying on MAPPO weakness for the main claim.",
            "remaining_action": "Run a MAPPO hyperparameter/action-head rescue study if the paper claims MAPPO-family superiority.",
        },
        {
            "issue_id": "R4",
            "reviewer_concern": "Controller-level QMIX/MAT are not full vehicle/RSU MARL baselines.",
            "status": "resolved_by_labeling",
            "evidence": "Protocol matrix labels MAPPO/QMIX/Controller-MAT as controller-level comparators.",
            "paper_resolution": "Use controller-level wording throughout; do not claim full vehicle-agent or RSU-agent MARL comparison.",
            "remaining_action": "Future work: freeze full multi-agent observation/action schema before adding full MARL baselines.",
        },
        {
            "issue_id": "R5",
            "reviewer_concern": "Hand-written Popularity-Cache is very close to SA-GHMAPPO.",
            "status": "limitation",
            "evidence": limitations.get("heuristic boundary", {}).get("evidence", "Popularity-Cache margin is small."),
            "paper_resolution": "Report heuristic rows as supplementary references and avoid large-gap claims against heuristics.",
            "remaining_action": "Add scenario-conditioned analysis showing when learned coordination beats the heuristic.",
        },
        {
            "issue_id": "R6",
            "reviewer_concern": "Offset holdout may overlap with formal windows and is not an external generalization test.",
            "status": "limitation",
            "evidence": "Current holdout is window-rank-offset validation, not a different dataset or guaranteed non-overlap split.",
            "paper_resolution": "Call it offset-window holdout; do not call it independent external holdout.",
            "remaining_action": "Add non-overlap temporal split or external highD/LuST validation before claiming broad generalization.",
        },
        {
            "issue_id": "R7",
            "reviewer_concern": "Prediction robustness is not universal.",
            "status": "resolved_by_claim_scope",
            "evidence": limitations.get("prediction boundary", {}).get("evidence", "no_prediction/oracle are mixed."),
            "paper_resolution": "Restrict wording to prediction-aware or calibrated surrogate-feature assistance until a learned predictor checkpoint is attached.",
            "remaining_action": "Treat no_prediction and oracle_prediction as diagnostics; do not call the current baseline predictor learned.",
        },
        {
            "issue_id": "R8",
            "reviewer_concern": "Not every system metric improves, especially backhaul and mechanism realization.",
            "status": "resolved_by_claim_scope",
            "evidence": (
                f"{limitations.get('mechanism boundary', {}).get('evidence', '')}; "
                f"{limitations.get('backhaul boundary', {}).get('evidence', '')}"
            ),
            "paper_resolution": "Use total reward, continuity, handoff failure, and migration overhead as the primary story; present backhaul/mechanism metrics as mixed diagnostics.",
            "remaining_action": "Avoid all-metric dominance language.",
        },
        {
            "issue_id": "R9",
            "reviewer_concern": "IPPO/DDQN-family rows may be cherry-picked or omitted.",
            "status": "resolved_by_exclusion_rule",
            "evidence": "IPPO is contract-blocked; DDQN and Dueling-DDQN require duplicate-trace independence audit before citation.",
            "paper_resolution": "Do not include these algorithms in the main table until they pass the same audit.",
            "remaining_action": "Optional: run duplicate-trace independent DDQN-family supplement.",
        },
    ]


def current_artifact_status(protocol_row: dict[str, Any]) -> str:
    role = str(protocol_row.get("paper_grade_role", ""))
    included = protocol_row.get("included_in_primary_gate") in {True, "True"}
    contract_status = str(protocol_row.get("contract_status", ""))
    if role == "candidate":
        return "main_method"
    if included and contract_status == "valid":
        return "primary_claimable"
    if included:
        return "primary_blocked_by_protocol"
    if protocol_row.get("role") == "optional_dqn_family_variant":
        return "optional_needs_independence_audit"
    if role == "not_in_current_primary_gate":
        return "available_needs_final_loop"
    if role == "supplementary_reference":
        return "supplementary_only"
    if role == "excluded":
        return "excluded"
    return "not_primary"


def training_budget_policy_for(protocol_row: dict[str, Any], status: str) -> str:
    role = str(protocol_row.get("role", ""))
    if status in {"supplementary_only", "excluded"}:
        return "not_trained"
    if role == "optional_dqn_family_variant":
        return "same_budget_after_duplicate_trace_audit"
    if status == "available_needs_final_loop":
        return "same_budget_required_in_new_final_loop"
    if status == "primary_blocked_by_protocol":
        return "matched_budget_but_protocol_blocked"
    if protocol_row.get("trainable_now") in {True, "True"}:
        return "matched_environment_interaction_budget"
    return "not_trained"


def build_algorithm_comparison_rows(protocol_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol_row in protocol_rows:
        agent_name = str(protocol_row.get("agent_name", ""))
        metadata = ALGORITHM_COMPARISON_METADATA.get(agent_name, {})
        status = current_artifact_status(protocol_row)
        if status == "primary_claimable":
            claim_boundary = "claimable in current artifact"
            if agent_name == "mappo":
                claim_boundary = "claimable as controller-level CTDE; underperformance must be reported with action-mix audit"
        elif status == "primary_blocked_by_protocol":
            claim_boundary = "listed in artifact but not claimable under current protocol"
        elif status == "available_needs_final_loop":
            claim_boundary = "implemented; needs final-submission rerun before citation"
        elif status == "optional_needs_independence_audit":
            claim_boundary = "optional only after duplicate-trace independence audit"
        elif status == "supplementary_only":
            claim_boundary = "supplementary reference only"
        elif status == "excluded":
            claim_boundary = "excluded from paper-grade learned baseline set"
        elif status == "main_method":
            claim_boundary = "candidate method"
        else:
            claim_boundary = "not a primary comparator in this artifact"
        rows.append(
            {
                "agent_name": agent_name,
                "display_name": agent_label(agent_name),
                "comparison_role": protocol_row.get("role", ""),
                "paper_grade_role": protocol_row.get("paper_grade_role", ""),
                "algorithm_family": metadata.get("family", "unspecified"),
                "learning_type": metadata.get("learning_type", "unspecified"),
                "contract_granularity": metadata.get("contract_granularity", "unspecified"),
                "uses_sa_only_mechanisms": metadata.get("uses_sa_only_mechanisms", "unknown"),
                "training_budget_policy": training_budget_policy_for(protocol_row, status),
                "current_artifact_status": status,
                "contract_status": protocol_row.get("contract_status", ""),
                "protocol_version": protocol_row.get("protocol_version", ""),
                "main_difference_from_sa": metadata.get("main_difference", ""),
                "literature_anchor": metadata.get("literature_anchor", ""),
                "claim_boundary": claim_boundary,
                "comparison_note": protocol_row.get("comparison_note", ""),
            }
        )
    return rows


def build_paper_paired_reward_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in paired_rows:
        if row.get("metric") != "total_reward" or row.get("baseline_agent") not in DEFAULT_LEARNED_BASELINES:
            continue
        output_rows.append(
            {
                "split": suite_display(str(row.get("suite", ""))),
                "suite": row.get("suite"),
                "baseline": agent_label(str(row.get("baseline_agent", ""))),
                "baseline_agent": row.get("baseline_agent"),
                "paired_count": row.get("paired_count"),
                "cluster_count": row.get("cluster_count"),
                "mean_delta": round_value(row.get("mean_delta")),
                "ci95_low": round_value(row.get("ci95_low")),
                "ci95_high": round_value(row.get("ci95_high")),
                "wins_ties_losses": wins_ties_losses(row),
                "sign_test_pvalue": round_value(row.get("sign_test_pvalue")),
                "claim_evidence": row.get("claim_evidence"),
            }
        )
    return output_rows


def build_paper_support_reward_rows(support_stat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in support_stat_rows:
        if (
            row.get("setting_id") != "all"
            or row.get("metric") != "total_reward"
            or row.get("baseline_agent") not in DEFAULT_LEARNED_BASELINES
        ):
            continue
        output_rows.append(
            {
                "support_suite": support_display(str(row.get("support_kind", ""))),
                "support_kind": row.get("support_kind"),
                "baseline": agent_label(str(row.get("baseline_agent", ""))),
                "baseline_agent": row.get("baseline_agent"),
                "paired_count": row.get("paired_count"),
                "mean_delta": round_value(row.get("mean_delta")),
                "ci95_low": round_value(row.get("ci95_low")),
                "ci95_high": round_value(row.get("ci95_high")),
                "wins_ties_losses": wins_ties_losses(row),
                "claim_evidence": row.get("claim_evidence"),
            }
        )
    return output_rows


def build_paper_prediction_setting_rows(
    support_stat_rows: list[dict[str, Any]],
    required_settings: list[str],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in support_stat_rows:
        if (
            row.get("support_kind") != "prediction"
            or row.get("setting_id") == "all"
            or row.get("metric") != "total_reward"
            or row.get("baseline_agent") != "ppo"
        ):
            continue
        setting_id = str(row.get("setting_id", ""))
        output_rows.append(
            {
                "prediction_setting": setting_id,
                "baseline": "PPO",
                "paired_count": row.get("paired_count"),
                "mean_delta": round_value(row.get("mean_delta")),
                "ci95_low": round_value(row.get("ci95_low")),
                "ci95_high": round_value(row.get("ci95_high")),
                "claim_scope": "claim_checked" if setting_id in required_settings else "diagnostic",
                "claim_evidence": row.get("claim_evidence"),
            }
        )
    return output_rows


def lookup_check(claim_checks: list[dict[str, Any]], check_name: str) -> dict[str, Any] | None:
    for check in claim_checks:
        if check.get("check") == check_name:
            return check
    return None


def build_self_review_rows(
    *,
    claim_checks: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    support_stat_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    mappo_action_audit_rows: list[dict[str, Any]],
    required_prediction_settings: list[str],
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []

    def add(category: str, status: str, severity: str, finding: str, evidence: str, paper_action: str) -> None:
        review_rows.append(
            {
                "category": category,
                "status": status,
                "severity": severity,
                "finding": finding,
                "evidence": evidence,
                "paper_action": paper_action,
            }
        )

    final_check = lookup_check(claim_checks, "final_gate_paper_claim_ready")
    add(
        "gate",
        "pass" if final_check and final_check.get("passed") else "block",
        "critical",
        "Final submission gate is paper-claim ready.",
        str(final_check.get("detail") if final_check else "missing"),
        "May cite this run as the canonical clean-retrain comparison only if this remains pass.",
    )
    provenance_check = lookup_check(claim_checks, "formal_training_provenance")
    add(
        "provenance",
        "pass" if provenance_check and provenance_check.get("passed") else "block",
        "critical",
        "Learned baselines are trained by the final suite or resumed from the same run.",
        str(provenance_check.get("detail") if provenance_check else "missing"),
        "State clean retraining and seed provenance in the experimental setup.",
    )
    for suite_name in ("formal_offset_0", "holdout_offset_3"):
        contract_check = lookup_check(claim_checks, f"{suite_name}_contract_and_independence")
        add(
            "baseline integrity",
            "pass" if contract_check and contract_check.get("passed") else "block",
            "critical",
            f"{suite_display(suite_name)} learned baselines pass contract and duplicate-trace audits.",
            str(contract_check.get("detail") if contract_check else "missing"),
            "Do not add diagnostic baselines to the main table unless the same audit passes.",
        )
        reward_check = lookup_check(claim_checks, f"{suite_name}_reward_ci_vs_all_learned")
        add(
            "reward significance",
            "pass" if reward_check and reward_check.get("passed") else "block",
            "critical",
            f"{suite_display(suite_name)} total-reward CIs against all learned baselines are strictly positive.",
            str(reward_check.get("detail") if reward_check else "missing"),
            "Use the strongest learned baseline in each split as the primary claim reference.",
        )

    min_learned_margin = min(
        float(row["sa_minus_strongest_learned_reward"])
        for row in margin_rows
        if safe_float(row.get("sa_minus_strongest_learned_reward")) is not None
    )
    add(
        "main table",
        "pass" if min_learned_margin > 0.0 else "block",
        "critical",
        "SA-GHMAPPO has a positive reward margin over the strongest learned baseline in every main split.",
        f"minimum margin={min_learned_margin:.6f}",
        "Main table can emphasize consistent learned-baseline dominance across formal and holdout splits.",
    )

    heuristic_margins = [
        float(row["sa_minus_strongest_heuristic_reward"])
        for row in margin_rows
        if safe_float(row.get("sa_minus_strongest_heuristic_reward")) is not None
    ]
    min_heuristic_margin = min(heuristic_margins) if heuristic_margins else 0.0
    add(
        "heuristic boundary",
        "limitation" if 0.0 <= min_heuristic_margin < 0.5 else "pass",
        "medium",
        "The hand-written popularity-cache reference is very close to SA-GHMAPPO.",
        f"minimum SA minus strongest heuristic reward margin={min_heuristic_margin:.6f}",
        "Report heuristics as supplementary references; avoid claiming a large heuristic gap.",
    )

    protocol_by_agent = {row.get("agent_name"): row for row in protocol_rows}
    ippo_row = protocol_by_agent.get("ippo", {})
    mappo_row = protocol_by_agent.get("mappo", {})
    qmix_row = protocol_by_agent.get("qmix", {})
    mat_row = protocol_by_agent.get("controller_mat", {})
    ippo_excluded_ok = bool(
        ippo_row
        and ippo_row.get("included_in_primary_gate") in {False, "False"}
        and ippo_row.get("contract_status") == "contract_blocked"
    )
    mappo_primary = bool(mappo_row.get("included_in_primary_gate") in {True, "True"})
    mappo_contract_ok = bool(
        mappo_primary
        and mappo_row.get("contract_status") == "valid"
        or (
            mappo_row
            and not mappo_primary
            and mappo_row.get("contract_status") in {
                "valid_controller_level_ctde",
                "valid_controller_level_ctde_head_credit",
            }
        )
    )
    add(
        "MAPPO/IPPO integrity",
        "pass"
        if ippo_excluded_ok and mappo_primary and mappo_contract_ok
        else "limitation"
        if ippo_excluded_ok and mappo_contract_ok
        else "block",
        "critical",
        "IPPO remains excluded; controller-level MAPPO must be included with the current head-credit v3 protocol or explicitly marked as not covered by this artifact.",
        (
            f"ippo:{ippo_row.get('contract_status')}, primary={ippo_row.get('included_in_primary_gate')}; "
            f"mappo:{mappo_row.get('contract_status')}, primary={mappo_row.get('included_in_primary_gate')}, "
            f"protocol={mappo_row.get('protocol_version', '')}"
        ),
        "Do not cite MAPPO results unless the artifact includes mappo, duplicate-trace audit passes, and the checkpoint protocol records controller head-credit v3.",
    )
    high_risk_mappo_rows = [
        row
        for row in mappo_action_audit_rows
        if row.get("risk_level") == "high"
        and row.get("reference_agent") in MAPPO_RISK_REFERENCE_AGENTS
    ]
    ppo_mappo_rows = [
        row
        for row in mappo_action_audit_rows
        if row.get("reference_agent") == "ppo"
    ]
    audit_reference_rows = ppo_mappo_rows or mappo_action_audit_rows
    worst_mappo_row = (
        min(
            audit_reference_rows,
            key=lambda row: safe_float(row.get("mappo_minus_reference_reward"))
            if safe_float(row.get("mappo_minus_reference_reward")) is not None
            else float("inf"),
        )
        if audit_reference_rows
        else {}
    )
    if mappo_primary and mappo_contract_ok and not mappo_action_audit_rows:
        mappo_action_status = "block"
        mappo_action_severity = "critical"
        mappo_action_finding = "MAPPO is a primary baseline but no action-mix audit was generated."
        mappo_action_evidence = "missing mappo_action_mix_audit rows"
        mappo_action_paper = "Regenerate the comparison report from benchmark_rows.csv before citing MAPPO."
    elif mappo_primary and mappo_contract_ok and high_risk_mappo_rows:
        mappo_action_status = "limitation"
        mappo_action_severity = "high"
        mappo_action_finding = (
            "MAPPO underperforms flat PPO/DQN because the controller-level action mix collapses away "
            "from prefetch/current-RSU execution."
        )
        mappo_action_evidence = (
            f"worst={worst_mappo_row.get('split')}/{worst_mappo_row.get('window_protocol')} "
            f"vs {worst_mappo_row.get('reference_agent')}: "
            f"reward_delta={worst_mappo_row.get('mappo_minus_reference_reward')}, "
            f"continuity_delta={worst_mappo_row.get('mappo_minus_reference_continuity')}, "
            f"failure_delta={worst_mappo_row.get('mappo_minus_reference_handoff_failure')}, "
            f"prefetch={worst_mappo_row.get('mappo_prefetch_action_count')} vs "
            f"{worst_mappo_row.get('reference_prefetch_action_count')}, "
            f"local={worst_mappo_row.get('mappo_local_exec_count')} vs "
            f"{worst_mappo_row.get('reference_local_exec_count')}, "
            f"diagnosis={worst_mappo_row.get('diagnosis')}"
        )
        mappo_action_paper = (
            "Include the MAPPO action-mix audit, describe MAPPO as a controller-level CTDE comparator, "
            "and make the strongest learned-baseline claim against PPO rather than relying on MAPPO weakness."
        )
    elif mappo_primary and mappo_contract_ok:
        mappo_action_status = "pass"
        mappo_action_severity = "medium"
        mappo_action_finding = "MAPPO action-mix diagnostics do not show a collapse relative to PPO/DQN."
        mappo_action_evidence = f"audit_rows={len(mappo_action_audit_rows)}"
        mappo_action_paper = "MAPPO can be reported under the normal controller-level CTDE baseline boundary."
    else:
        mappo_action_status = "limitation"
        mappo_action_severity = "medium"
        mappo_action_finding = "MAPPO action-mix audit is only relevant once MAPPO is included as a valid primary baseline."
        mappo_action_evidence = f"mappo_primary={mappo_primary}, mappo_contract_ok={mappo_contract_ok}"
        mappo_action_paper = "Do not use MAPPO numeric results outside artifacts where MAPPO is primary and protocol-valid."
    add(
        "MAPPO action-mix risk",
        mappo_action_status,
        mappo_action_severity,
        mappo_action_finding,
        mappo_action_evidence,
        mappo_action_paper,
    )
    qmix_primary = bool(qmix_row.get("included_in_primary_gate") in {True, "True"})
    qmix_contract_ok = bool(
        qmix_primary
        and qmix_row.get("contract_status") == "valid"
        or (
            qmix_row
            and not qmix_primary
            and qmix_row.get("contract_status") == "valid_controller_level_value_decomposition"
        )
    )
    add(
        "QMIX integrity",
        "pass" if qmix_primary else "limitation" if qmix_contract_ok else "block",
        "high",
        "Controller-level QMIX must be either included in a new primary gate or explicitly marked as not covered by this artifact.",
        f"qmix:{qmix_row.get('contract_status')}, primary={qmix_row.get('included_in_primary_gate')}",
        "Do not cite QMIX results unless the current artifact learned_baseline_agents includes qmix and its duplicate-trace audit passes.",
    )
    mat_primary = bool(mat_row.get("included_in_primary_gate") in {True, "True"})
    mat_contract_ok = bool(
        mat_primary
        and mat_row.get("contract_status") == "valid"
        or (
            mat_row
            and not mat_primary
            and mat_row.get("contract_status") == "valid_controller_level_transformer_ctde"
        )
    )
    add(
        "Controller-MAT integrity",
        "pass" if mat_primary else "limitation" if mat_contract_ok else "block",
        "high",
        "Controller-level transformer MARL must be either included in a new primary gate or explicitly marked as not covered by this artifact.",
        f"controller_mat:{mat_row.get('contract_status')}, primary={mat_row.get('included_in_primary_gate')}",
        "Do not cite Controller-MAT results unless the current artifact learned_baseline_agents includes controller_mat and its duplicate-trace audit passes.",
    )
    domain_statuses = {
        "dag_offload_drl": "valid_domain_dag_offload",
        "cache_offload_drl": "valid_domain_model_cache_offload",
        "dt_handoff_drl": "valid_domain_digital_twin_handoff",
    }
    domain_primary_count = 0
    domain_contract_ok_count = 0
    domain_evidence: list[str] = []
    for agent_name, optional_status in domain_statuses.items():
        row = protocol_by_agent.get(agent_name, {})
        is_primary = bool(row.get("included_in_primary_gate") in {True, "True"})
        contract_ok = bool(
            is_primary
            and row.get("contract_status") == "valid"
            or (row and not is_primary and row.get("contract_status") == optional_status)
        )
        domain_primary_count += 1 if is_primary else 0
        domain_contract_ok_count += 1 if contract_ok else 0
        domain_evidence.append(
            f"{agent_name}:{row.get('contract_status')}, primary={row.get('included_in_primary_gate')}"
        )
    add(
        "domain comparator integrity",
        "pass" if domain_primary_count == len(domain_statuses) else "limitation" if domain_contract_ok_count == len(domain_statuses) else "block",
        "high",
        "DAG, model-cache, and digital-twin handoff domain baselines must be included in a new primary gate or explicitly marked as not covered by this artifact.",
        "; ".join(domain_evidence),
        "Do not cite domain-baseline results unless the current artifact includes dag_offload_drl, cache_offload_drl, and dt_handoff_drl in learned_baseline_agents and duplicate-trace audit passes.",
    )

    primary_learned_agents = [
        str(row.get("agent_name"))
        for row in protocol_rows
        if row.get("agent_name") in DEFAULT_LEARNED_BASELINES
        and row.get("included_in_primary_gate") in {True, "True"}
    ] or list(DEFAULT_LEARNED_BASELINES)
    required_support_kinds = ("prediction", "robustness", "scalability")
    support_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in support_stat_rows:
        support_kind = str(row.get("support_kind", ""))
        baseline_agent = str(row.get("baseline_agent", ""))
        if (
            row.get("setting_id") == "all"
            and row.get("metric") == "total_reward"
            and support_kind in required_support_kinds
            and baseline_agent in primary_learned_agents
        ):
            support_rows_by_key[(support_kind, baseline_agent)] = row
    missing_support = [
        (support_kind, agent_name)
        for support_kind in required_support_kinds
        for agent_name in primary_learned_agents
        if (support_kind, agent_name) not in support_rows_by_key
    ]
    non_positive_support = [
        row
        for row in support_rows_by_key.values()
        if safe_float(row.get("ci95_low")) is None or float(row["ci95_low"]) <= 0.0
    ]
    support_evidence_parts: list[str] = []
    for support_kind in required_support_kinds:
        present_rows = [
            support_rows_by_key[(support_kind, agent_name)]
            for agent_name in primary_learned_agents
            if (support_kind, agent_name) in support_rows_by_key
        ]
        if not present_rows:
            continue
        weakest_row = min(
            present_rows,
            key=lambda item: safe_float(item.get("ci95_low"))
            if safe_float(item.get("ci95_low")) is not None
            else float("-inf"),
        )
        support_evidence_parts.append(
            f"{support_kind}: {len(present_rows)}/{len(primary_learned_agents)} present, "
            f"weakest vs {weakest_row.get('baseline_agent')} mean={weakest_row.get('mean_delta')} "
            f"ci=[{weakest_row.get('ci95_low')},{weakest_row.get('ci95_high')}]"
        )
    if missing_support:
        missing_text = ", ".join(f"{kind}/{agent}" for kind, agent in missing_support[:8])
        support_evidence_parts.append(
            f"missing={missing_text}" + ("..." if len(missing_support) > 8 else "")
        )
    if non_positive_support:
        non_positive_text = ", ".join(
            f"{row.get('support_kind')}/{row.get('baseline_agent')} ci_low={row.get('ci95_low')}"
            for row in non_positive_support[:8]
        )
        support_evidence_parts.append(
            f"non_positive={non_positive_text}" + ("..." if len(non_positive_support) > 8 else "")
        )
    support_pass = not missing_support and not non_positive_support
    add(
        "support suites",
        "pass" if support_pass else "block",
        "high",
        "Prediction, robustness, and scalability aggregate reward CIs against all primary learned baselines are positive.",
        "; ".join(support_evidence_parts),
        "Use these as support evidence after the main formal/holdout table.",
    )

    required_prediction_rows = [
        row
        for row in support_stat_rows
        if row.get("support_kind") == "prediction"
        and row.get("setting_id") in required_prediction_settings
        and row.get("metric") == "total_reward"
        and row.get("baseline_agent") == "ppo"
    ]
    required_prediction_pass = bool(required_prediction_rows) and all(
        float(row["ci95_low"]) > 0.0 for row in required_prediction_rows
    )
    add(
        "prediction claim",
        "pass" if required_prediction_pass else "block",
        "high",
        "Claim-relevant calibrated/noisy prediction-aware settings beat PPO with positive reward CIs.",
        "; ".join(
            f"{row.get('setting_id')} mean={row.get('mean_delta')} ci=[{row.get('ci95_low')},{row.get('ci95_high')}]"
            for row in required_prediction_rows
        ),
        "Limit wording to prediction-aware or calibrated surrogate-feature assistance unless a learned predictor checkpoint is attached.",
    )

    diagnostic_prediction_rows = [
        row
        for row in support_stat_rows
        if row.get("support_kind") == "prediction"
        and row.get("setting_id") in {"no_prediction", "oracle_prediction"}
        and row.get("metric") == "total_reward"
        and row.get("baseline_agent") == "ppo"
        and safe_float(row.get("ci95_low")) is not None
        and float(row["ci95_low"]) <= 0.0
    ]
    add(
        "prediction boundary",
        "limitation" if diagnostic_prediction_rows else "pass",
        "medium",
        "Diagnostic no-prediction/oracle settings and current baseline predictors do not support a learned-surrogate claim.",
        "; ".join(
            f"{row.get('setting_id')} mean={row.get('mean_delta')} ci=[{row.get('ci95_low')},{row.get('ci95_high')}]"
            for row in diagnostic_prediction_rows
        )
        or "all diagnostic settings positive",
        "Keep no_prediction and oracle_prediction as stress diagnostics; treat baseline/calibrated predictors as feature assistance, not learned surrogate evidence.",
    )

    mechanism_weak_rows = [
        row
        for row in paired_rows
        if row.get("baseline_agent") == "ppo"
        and row.get("metric") == "mechanism_realization_rate"
        and row.get("claim_evidence") != "positive_ci"
    ]
    add(
        "mechanism boundary",
        "limitation" if mechanism_weak_rows else "pass",
        "medium",
        "Mechanism realization rate is not a standalone positive-CI advantage over PPO in every split.",
        "; ".join(
            f"{row.get('suite')} mean={row.get('mean_delta')} ci=[{row.get('ci95_low')},{row.get('ci95_high')}]"
            for row in mechanism_weak_rows
        )
        or "positive across splits",
        "Emphasize reward, continuity, handoff failure, and migration overhead; avoid claiming all mechanism metrics improve.",
    )

    backhaul_weak_rows = [
        row
        for row in paired_rows
        if row.get("baseline_agent") == "ppo"
        and row.get("metric") == "backhaul_traffic_cost"
        and row.get("claim_evidence") != "positive_ci"
    ]
    add(
        "backhaul boundary",
        "limitation" if backhaul_weak_rows else "pass",
        "medium",
        "Backhaul savings against PPO are not positive-CI in every split.",
        "; ".join(
            f"{row.get('suite')} mean={row.get('mean_delta')} ci=[{row.get('ci95_low')},{row.get('ci95_high')}]"
            for row in backhaul_weak_rows
        )
        or "positive across splits",
        "Treat backhaul as supportive where significant, not as a universal headline result.",
    )

    return review_rows


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if "\\" in text or "$" in text:
        return text
    return latex_escape(text)


def latex_bold(value: str) -> str:
    return rf"\textbf{{{value}}}"


def write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    column_format: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if column_format is None:
        column_format = "l" * len(columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_cell(row.get(column, "")) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_latex_main_rows(rows: list[dict[str, Any]], table_agents: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output_row: dict[str, Any] = {
            "Split": row.get("split"),
            "Mode": row.get("window_protocol"),
            "N": row.get("episode_count"),
        }
        for agent_name in table_agents:
            label = agent_label(agent_name)
            value_text = format_num(row.get(f"{agent_name}_reward"))
            output_row[label] = latex_bold(value_text) if agent_name == "sa_ghmappo" else value_text
        output_row["Delta vs PPO"] = latex_bold(signed_text(row.get("delta_vs_ppo")))
        output_row["Rel."] = latex_bold(percent_text(row.get("relative_delta_vs_ppo_percent"))).replace("%", r"\%")
        output_row["Delta vs Pop."] = signed_text(row.get("delta_vs_strongest_heuristic_reference"))
        output.append(output_row)
    return output


def build_latex_algorithm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "Comparator": row.get("display_name"),
                "Role": LATEX_ROLE_LABELS.get(str(row.get("paper_grade_role", "")), row.get("paper_grade_role")),
                "Family": row.get("algorithm_family"),
                "Contract": row.get("contract_granularity"),
                "Budget": LATEX_BUDGET_LABELS.get(
                    str(row.get("training_budget_policy", "")),
                    row.get("training_budget_policy"),
                ),
                "Status": LATEX_STATUS_LABELS.get(
                    str(row.get("current_artifact_status", "")),
                    row.get("current_artifact_status"),
                ),
                "Boundary": row.get("claim_boundary"),
            }
        )
    return output


def build_latex_paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "Split": row.get("split"),
                "Baseline": row.get("baseline"),
                "Pairs": row.get("paired_count"),
                "Clusters": row.get("cluster_count"),
                "Mean Delta": latex_bold(signed_text(row.get("mean_delta"))),
                "95% CI": latex_bold(f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]"),
                "W/T/L": row.get("wins_ties_losses"),
                "p": format_num(row.get("sign_test_pvalue"), 6),
            }
        )
    return output


def build_latex_support_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "Suite": row.get("support_suite"),
                "Baseline": row.get("baseline"),
                "Pairs": row.get("paired_count"),
                "Mean Delta": latex_bold(signed_text(row.get("mean_delta"))),
                "95% CI": latex_bold(f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]"),
                "Evidence": row.get("claim_evidence"),
            }
        )
    return output


def build_latex_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "Setting": row.get("prediction_setting"),
                "Scope": row.get("claim_scope"),
                "Pairs": row.get("paired_count"),
                "Mean Delta": signed_text(row.get("mean_delta")),
                "95% CI": f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]",
                "Evidence": row.get("claim_evidence"),
            }
        )
    return output


def build_latex_mappo_action_mix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "Split": row.get("split"),
                "Mode": row.get("window_protocol"),
                "Reference": row.get("reference"),
                "Reward Delta": signed_text(row.get("mappo_minus_reference_reward")),
                "Continuity Delta": signed_text(row.get("mappo_minus_reference_continuity")),
                "Failure Delta": signed_text(row.get("mappo_minus_reference_handoff_failure")),
                "MAPPO Prefetch": format_num(row.get("mappo_prefetch_action_count")),
                "Ref Prefetch": format_num(row.get("reference_prefetch_action_count")),
                "MAPPO Local": format_num(row.get("mappo_local_exec_count")),
                "Ref Local": format_num(row.get("reference_local_exec_count")),
                "Risk": row.get("risk_level"),
            }
        )
    return output


def build_latex_audited_main_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        is_main = row.get("agent_name") == "sa_ghmappo"
        output.append(
            {
                "Algorithm": row.get("display_name"),
                "Role": AUDITED_ROLE_LABELS.get(
                    str(row.get("paper_table_role", "")),
                    row.get("paper_table_role"),
                ),
                "Audit": AUDIT_DECISION_LABELS.get(
                    str(row.get("audit_decision", "")),
                    row.get("audit_decision"),
                ),
                "Formal-M": latex_bold(format_num(row.get("formal_mixed_reward"))) if is_main else format_num(row.get("formal_mixed_reward")),
                "Formal-F": latex_bold(format_num(row.get("formal_full_reward"))) if is_main else format_num(row.get("formal_full_reward")),
                "Holdout-M": latex_bold(format_num(row.get("holdout_mixed_reward"))) if is_main else format_num(row.get("holdout_mixed_reward")),
                "Holdout-F": latex_bold(format_num(row.get("holdout_full_reward"))) if is_main else format_num(row.get("holdout_full_reward")),
                "Formal Delta CI": row.get("formal_sa_minus_agent_reward_ci"),
                "Holdout Delta CI": row.get("holdout_sa_minus_agent_reward_ci"),
            }
        )
    return output


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_markdown_audited_main_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "Algorithm": row.get("display_name"),
                "Role": AUDITED_ROLE_LABELS.get(
                    str(row.get("paper_table_role", "")),
                    row.get("paper_table_role"),
                ),
                "Audit": AUDIT_DECISION_LABELS.get(
                    str(row.get("audit_decision", "")),
                    row.get("audit_decision"),
                ),
                "Formal Mixed": format_num(row.get("formal_mixed_reward")),
                "Formal Full": format_num(row.get("formal_full_reward")),
                "Holdout Mixed": format_num(row.get("holdout_mixed_reward")),
                "Holdout Full": format_num(row.get("holdout_full_reward")),
                "Formal CI": row.get("formal_sa_minus_agent_reward_ci"),
                "Holdout CI": row.get("holdout_sa_minus_agent_reward_ci"),
                "Boundary": row.get("claim_boundary"),
            }
        )
    return output


def build_latex_fairness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Algorithm": row.get("display_name"),
            "Seeds": row.get("seeds"),
            "Episodes": row.get("episodes"),
            "Updates": row.get("update_count"),
            "Obs.": row.get("observation_contract"),
            "Action": row.get("action_contract"),
            "Status": row.get("audit_status"),
        }
        for row in rows
    ]


def build_latex_strongest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Split": row.get("split"),
            "Mode": row.get("window_protocol"),
            "Strongest Learned": row.get("strongest_learned"),
            "Reward": format_num(row.get("strongest_learned_reward")),
            "PPO Rank": row.get("ppo_rank_among_learned"),
            "Top-3": row.get("top3_learned_by_reward"),
            "Strongest Heuristic": row.get("strongest_heuristic"),
        }
        for row in rows
    ]


def build_latex_issue_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ID": row.get("issue_id"),
            "Concern": row.get("reviewer_concern"),
            "Status": row.get("status"),
            "Resolution": row.get("paper_resolution"),
        }
        for row in rows
    ]


def build_paper_result_summary(
    *,
    final_run_id: str,
    learned_baseline_agents: list[str],
    protocol_rows: list[dict[str, Any]],
    package_ready: bool,
    paper_main_rows: list[dict[str, Any]],
    paper_paired_rows: list[dict[str, Any]],
    paper_support_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    mappo_action_audit_rows: list[dict[str, Any]],
    self_review_rows: list[dict[str, Any]],
) -> str:
    protocol_by_agent = {str(row.get("agent_name", "")): row for row in protocol_rows}
    claimable_learned_agents = [
        agent_name
        for agent_name in learned_baseline_agents
        if protocol_by_agent.get(agent_name, {}).get("included_in_primary_gate") in {True, "True"}
        and protocol_by_agent.get(agent_name, {}).get("contract_status") == "valid"
    ]
    statement_agents = learned_baseline_agents if package_ready else claimable_learned_agents
    if not statement_agents:
        statement_agents = learned_baseline_agents
    learned_text = ", ".join(f"`{agent_name}`" for agent_name in learned_baseline_agents)
    claimable_text = ", ".join(f"`{agent_name}`" for agent_name in claimable_learned_agents) or "`none`"
    primary_mappo_included = "mappo" in learned_baseline_agents
    mappo_row = protocol_by_agent.get("mappo", {})
    mappo_contract_status = str(mappo_row.get("contract_status", ""))
    mappo_protocol_version = str(mappo_row.get("protocol_version", ""))
    if primary_mappo_included and mappo_contract_status == "valid":
        mappo_status_line = "- MAPPO status: included as a controller-level CTDE baseline with controller head-credit v3"
    elif primary_mappo_included:
        mappo_status_line = (
            "- MAPPO status: numeric row is present but blocked for paper claim "
            f"(`{mappo_contract_status}`, protocol=`{mappo_protocol_version}`)"
        )
    else:
        mappo_status_line = "- MAPPO status: implemented as controller-level CTDE, but not covered by this artifact"
    mappo_high_risk_rows = [
        row for row in mappo_action_audit_rows if row.get("risk_level") == "high"
    ]
    if primary_mappo_included and mappo_contract_status == "valid" and mappo_action_audit_rows:
        mappo_action_status_line = (
            "- MAPPO action-mix audit: generated and reviewer-facing "
            f"(`{len(mappo_high_risk_rows)}` high-risk rows out of `{len(mappo_action_audit_rows)}` comparisons)"
        )
    elif primary_mappo_included and mappo_contract_status == "valid":
        mappo_action_status_line = "- MAPPO action-mix audit: missing; regenerate before citation"
    else:
        mappo_action_status_line = "- MAPPO action-mix audit: not required for this artifact scope"

    margin_values: list[float] = []
    for row in paper_main_rows:
        sa_reward = safe_float(row.get("sa_ghmappo_reward"))
        if sa_reward is None:
            continue
        baseline_rewards = [
            float(value)
            for agent_name in statement_agents
            if (value := safe_float(row.get(f"{agent_name}_reward"))) is not None
        ]
        if baseline_rewards:
            margin_values.append(float(sa_reward) - max(baseline_rewards))
    if not margin_values:
        margin_values = [
            float(row["delta_vs_strongest_learned"])
            for row in paper_main_rows
            if safe_float(row.get("delta_vs_strongest_learned")) is not None
        ]
    min_delta = min(margin_values)
    max_delta = max(margin_values)

    def weakest_positive_stat(suite_name: str) -> dict[str, Any]:
        candidates = [
            row
            for row in paper_paired_rows
            if row.get("suite") == suite_name
            and row.get("baseline_agent") in statement_agents
            and safe_float(row.get("mean_delta")) is not None
        ]
        return min(candidates, key=lambda row: float(row["mean_delta"]))

    formal_stat = weakest_positive_stat("formal_offset_0")
    holdout_stat = weakest_positive_stat("holdout_offset_3")
    formal_baseline = str(formal_stat.get("baseline_agent", ""))
    holdout_baseline = str(holdout_stat.get("baseline_agent", ""))
    support_summary_parts: list[str] = []
    for support_kind in ("prediction", "robustness", "scalability"):
        kind_rows = [
            row
            for row in paper_support_rows
            if row.get("support_kind") == support_kind
            and row.get("baseline_agent") in statement_agents
            and safe_float(row.get("ci95_low")) is not None
        ]
        if not kind_rows:
            continue
        weakest_row = min(kind_rows, key=lambda row: float(row["ci95_low"]))
        support_summary_parts.append(
            f"{support_display(support_kind)} weakest vs {agent_label(str(weakest_row['baseline_agent']))} "
            f"mean delta {float(weakest_row['mean_delta']):.3f} "
            f"(95% CI [{float(weakest_row['ci95_low']):.3f}, {float(weakest_row['ci95_high']):.3f}])"
        )
    support_statement = (
        "The support suites remain positive against the stated learned-baseline scope: "
        + "; ".join(support_summary_parts)
        + "."
        if support_summary_parts
        else "The support suite reward summary is unavailable in this package."
    )
    limitations = [row for row in self_review_rows if row.get("status") == "limitation"]
    blockers = [row for row in self_review_rows if row.get("status") == "block"]
    package_heading = "# Paper-Ready Comparison Package" if package_ready else "# Comparison Package Audit"
    result_heading = "## Copy-Ready Result Statement" if package_ready else "## Conditional Numeric Summary"
    result_scope = (
        "clean-retrained primary learned baselines"
        if package_ready
        else "clean-retrained claimable learned-baseline subset"
    )

    lines = [
        package_heading,
        "",
        f"- canonical_run: `{final_run_id}`",
        f"- paper_ready_package_ready: `{str(package_ready).lower()}`",
        f"- primary learned baselines: {learned_text}",
        f"- claimable learned baselines in this artifact: {claimable_text}",
        "- diagnostic exclusions: `ippo` remains contract-blocked under the current wrapper contract",
        mappo_status_line,
        mappo_action_status_line,
        f"- blocker_count: `{len(blockers)}`",
        "",
        result_heading,
        "",
    ]
    if not package_ready:
        lines.extend(
            [
                "This package has unresolved blocker rows in the self-review. Treat the numeric summary below as an audit aid, not as a paper-ready claim.",
                "",
            ]
        )
    lines.extend(
        [
        (
            "On the NGSIM + Alibaba benchmark, SA-GHMAPPO ranks first among the "
            f"{result_scope} in all formal and offset-3 holdout splits. Against the strongest learned baseline "
            f"in each split, the aggregate reward gain ranges from {min_delta:.3f} to {max_delta:.3f}. "
            f"Cluster-bootstrap paired tests remain positive in both the formal split "
            f"(vs {agent_label(formal_baseline)}, mean delta {float(formal_stat['mean_delta']):.3f}, 95% CI "
            f"[{float(formal_stat['ci95_low']):.3f}, {float(formal_stat['ci95_high']):.3f}]) and the "
            f"offset-3 holdout split (vs {agent_label(holdout_baseline)}, mean delta "
            f"{float(holdout_stat['mean_delta']):.3f}, 95% CI "
            f"[{float(holdout_stat['ci95_low']):.3f}, {float(holdout_stat['ci95_high']):.3f}])."
        ),
        "",
        support_statement,
        "",
        "## MAPPO Action-Mix Audit",
        "",
        "This diagnostic is included to avoid overstating MAPPO weakness. MAPPO is treated as a controller-level CTDE comparator; when it underperforms flat PPO/DQN, the paper should report the action-mix cause and keep the strongest-baseline claim anchored to PPO.",
        "",
        "## Author Self-Review",
        "",
        ]
    )
    if mappo_action_audit_rows:
        for row in mappo_action_audit_rows:
            if row.get("reference_agent") != "ppo":
                continue
            lines.insert(
                -3,
                (
                    f"- {row.get('split')}/{row.get('window_protocol')} vs PPO: "
                    f"reward delta {float(row['mappo_minus_reference_reward']):+.3f}, "
                    f"continuity delta {float(row['mappo_minus_reference_continuity']):+.3f}, "
                    f"failure delta {float(row['mappo_minus_reference_handoff_failure']):+.3f}; "
                    f"prefetch {float(row['mappo_prefetch_action_count']):.3f} vs "
                    f"{float(row['reference_prefetch_action_count']):.3f}, "
                    f"local {float(row['mappo_local_exec_count']):.3f} vs "
                    f"{float(row['reference_local_exec_count']):.3f}; "
                    f"{row.get('diagnosis')}"
                ),
            )
        lines.insert(-3, "")
    else:
        lines.insert(-3, "- No MAPPO action-mix rows were available.")
        lines.insert(-3, "")
    for row in self_review_rows:
        lines.append(
            f"- {row['status'].upper()} [{row['severity']}] {row['finding']} "
            f"Evidence: {row['evidence']} Action: {row['paper_action']}"
        )
    lines.extend(
        [
            "",
            "## Required Claim Boundary",
            "",
            "- Do not report IPPO as a formal baseline until a true independent per-agent wrapper is implemented.",
            "- Cite MAPPO only when the current artifact includes `mappo` in `learned_baseline_agents`, duplicate-trace audit passes, and `baseline_protocol_versions.mappo` records the current controller head-credit v3 protocol.",
            "- If MAPPO is weaker than PPO/DQN, include the action-mix audit and avoid framing MAPPO weakness as the main evidence for SA-GHMAPPO.",
            "- Cite Controller-MAT only when the current artifact includes `controller_mat` in `learned_baseline_agents` and duplicate-trace audit passes.",
            "- Cite DAG/cache/DT domain baselines only when the current artifact includes `dag_offload_drl`, `cache_offload_drl`, and `dt_handoff_drl` in `learned_baseline_agents` and duplicate-trace audit passes.",
            "- Do not claim large dominance over the hand-written popularity-cache heuristic; it is a close supplementary reference.",
            "- Do not claim a learned surrogate predictor unless a learned predictor checkpoint is attached; no_prediction and oracle_prediction are diagnostic stress cases.",
            "- Do not claim every mechanism metric improves; mechanism_realization_rate and some backhaul comparisons are mixed.",
            "",
            "## Prediction Setting Audit vs PPO",
            "",
        ]
    )
    for row in prediction_rows:
        lines.append(
            f"- `{row['prediction_setting']}` ({row['claim_scope']}): mean delta "
            f"{float(row['mean_delta']):+.3f}, 95% CI "
            f"[{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
            f"{row['claim_evidence']}"
        )
    if limitations:
        lines.extend(["", "## Reviewer-Facing Limitations", ""])
        for row in limitations:
            lines.append(f"- {row['finding']} ({row['paper_action']})")
    lines.append("")
    return "\n".join(lines)


def format_num(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def build_markdown(
    *,
    report: dict[str, Any],
    margin_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    support_stats: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    mappo_action_audit_rows: list[dict[str, Any]],
) -> str:
    review_ready = report["review_ready"]
    lines: list[str] = [
        "# Top-Journal Comparison Report",
        "",
        f"- final_run_id: `{report['final_run_id']}`",
        f"- review_ready: `{str(review_ready).lower()}`",
        f"- paper_ready_package_ready: `{str(report.get('paper_ready_package_ready', False)).lower()}`",
        f"- learned_baselines: `{' '.join(report['learned_baseline_agents'])}`",
        f"- heuristic_policy: `{report['heuristic_policy']}`",
        "",
        "## Paper-Ready Outputs",
        "",
        f"- main table: `{report['output_files'].get('paper_ready_main_comparison_tex')}`",
        f"- paired statistics: `{report['output_files'].get('paper_ready_paired_reward_statistics_tex')}`",
        f"- support statistics: `{report['output_files'].get('paper_ready_support_reward_statistics_tex')}`",
        f"- self-review: `{report['output_files'].get('paper_ready_report_md')}`",
        "",
        "## Main Reward Margins",
    ]
    margin_table = [
        {
            "suite": row.get("suite"),
            "mode": row.get("mode"),
            "SA": format_num(row.get("sa_total_reward")),
            "strongest learned": row.get("strongest_learned_baseline"),
            "baseline": format_num(row.get("strongest_learned_baseline_reward")),
            "delta": format_num(row.get("sa_minus_strongest_learned_reward")),
            "heuristic delta": format_num(row.get("sa_minus_strongest_heuristic_reward")),
        }
        for row in margin_rows
    ]
    lines.extend(markdown_table(margin_table, ["suite", "mode", "SA", "strongest learned", "baseline", "delta", "heuristic delta"]))
    lines.extend(["", "## Paired Reward Statistics"])
    reward_stat_table = [
        {
            "suite": row.get("suite"),
            "baseline": row.get("baseline_agent"),
            "mean delta": format_num(row.get("mean_delta")),
            "95% CI": f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]",
            "wins/ties/losses": f"{row.get('wins')}/{row.get('ties')}/{row.get('losses')}",
            "p": format_num(row.get("sign_test_pvalue"), 6),
        }
        for row in paired_rows
        if row.get("metric") == "total_reward" and row.get("baseline_agent") in DEFAULT_LEARNED_BASELINES
    ]
    lines.extend(markdown_table(reward_stat_table, ["suite", "baseline", "mean delta", "95% CI", "wins/ties/losses", "p"]))
    lines.extend(["", "## Mechanism Decomposition vs PPO"])
    mechanism_table = [
        {
            "suite": row.get("suite"),
            "metric": row.get("metric"),
            "signed delta": format_num(row.get("mean_delta")),
            "95% CI": f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]",
            "evidence": row.get("claim_evidence"),
        }
        for row in paired_rows
        if row.get("baseline_agent") == "ppo" and row.get("metric") in DEFAULT_METRICS
    ]
    lines.extend(markdown_table(mechanism_table, ["suite", "metric", "signed delta", "95% CI", "evidence"]))
    lines.extend(["", "## Support Suite Reward Statistics"])
    support_table = [
        {
            "support": row.get("support_kind"),
            "baseline": row.get("baseline_agent"),
            "mean delta": format_num(row.get("mean_delta")),
            "95% CI": f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]",
            "evidence": row.get("claim_evidence"),
        }
        for row in support_stats
        if row.get("setting_id") == "all"
        and row.get("metric") == "total_reward"
        and row.get("baseline_agent") in DEFAULT_LEARNED_BASELINES
    ]
    lines.extend(markdown_table(support_table, ["support", "baseline", "mean delta", "95% CI", "evidence"]))
    lines.extend(["", "## Prediction Setting Reward Statistics"])
    prediction_setting_table = [
        {
            "setting": row.get("setting_id"),
            "baseline": row.get("baseline_agent"),
            "mean delta": format_num(row.get("mean_delta")),
            "95% CI": f"[{format_num(row.get('ci95_low'))}, {format_num(row.get('ci95_high'))}]",
            "evidence": row.get("claim_evidence"),
        }
        for row in support_stats
        if row.get("support_kind") == "prediction"
        and row.get("setting_id") != "all"
        and row.get("metric") == "total_reward"
        and row.get("baseline_agent") in DEFAULT_LEARNED_BASELINES
    ]
    lines.extend(
        markdown_table(prediction_setting_table, ["setting", "baseline", "mean delta", "95% CI", "evidence"])
    )
    lines.extend(["", "## Baseline Protocol Matrix"])
    protocol_table = [
        {
            "agent": row.get("agent_name"),
            "role": row.get("role"),
            "primary": row.get("included_in_primary_gate"),
            "contract": row.get("contract_status"),
            "protocol": row.get("protocol_version", ""),
        }
        for row in protocol_rows
    ]
    lines.extend(markdown_table(protocol_table, ["agent", "role", "primary", "contract", "protocol"]))
    lines.extend(["", "## Algorithm Comparison Table"])
    algorithm_rows = build_algorithm_comparison_rows(protocol_rows)
    algorithm_table = [
        {
            "agent": row.get("agent_name"),
            "family": row.get("algorithm_family"),
            "type": row.get("learning_type"),
            "anchor": row.get("literature_anchor"),
            "budget": row.get("training_budget_policy"),
            "status": row.get("current_artifact_status"),
            "boundary": row.get("claim_boundary"),
        }
        for row in algorithm_rows
    ]
    lines.extend(markdown_table(algorithm_table, ["agent", "family", "type", "anchor", "budget", "status", "boundary"]))
    lines.extend(["", "## MAPPO Action-Mix Audit"])
    mappo_action_table = [
        {
            "split": row.get("split"),
            "mode": row.get("window_protocol"),
            "reference": row.get("reference_agent"),
            "reward delta": format_num(row.get("mappo_minus_reference_reward")),
            "continuity delta": format_num(row.get("mappo_minus_reference_continuity")),
            "failure delta": format_num(row.get("mappo_minus_reference_handoff_failure")),
            "mappo prefetch": format_num(row.get("mappo_prefetch_action_count")),
            "ref prefetch": format_num(row.get("reference_prefetch_action_count")),
            "mappo local": format_num(row.get("mappo_local_exec_count")),
            "ref local": format_num(row.get("reference_local_exec_count")),
            "risk": row.get("risk_level"),
            "diagnosis": row.get("diagnosis"),
        }
        for row in mappo_action_audit_rows
    ]
    lines.extend(
        markdown_table(
            mappo_action_table,
            [
                "split",
                "mode",
                "reference",
                "reward delta",
                "continuity delta",
                "failure delta",
                "mappo prefetch",
                "ref prefetch",
                "mappo local",
                "ref local",
                "risk",
                "diagnosis",
            ],
        )
    )
    lines.extend(["", "## Claim Boundary"])
    for item in report["claim_boundary"]:
        lines.append(f"- {item}")
    if not review_ready:
        lines.extend(["", "## Blockers"])
        for check in report["claim_checks"]:
            if not check["passed"]:
                lines.append(f"- {check['check']}: {check['detail']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    final_root = args.final_run_root
    output_dir = args.output_dir or final_root / "comparison_report"
    final_gate_path = final_root / "final_submission_gate_report.json"
    final_gate = load_json(final_gate_path)
    rng = random.Random(args.random_seed)

    suite_reports = collect_suite_reports(final_gate, final_root)
    main_metric_rows = collect_main_metric_rows(suite_reports, final_root)
    action_mix_summary_rows = collect_action_mix_summary_rows(suite_reports, final_root)
    mappo_action_audit_rows = build_mappo_action_mix_audit_rows(action_mix_summary_rows)
    margin_rows = collect_margin_rows(suite_reports)
    paired_rows = load_existing_statistics(suite_reports, final_root)
    support_summary_rows, support_stat_rows = collect_support_rows(
        final_gate,
        final_root,
        candidate_agent=args.candidate_agent,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    protocol_rows = build_protocol_rows(final_gate)
    claim_checks = build_claim_checks(final_gate, suite_reports, margin_rows, paired_rows, support_stat_rows)
    paper_output_dir = output_dir / "paper_ready"
    paper_table_agents = unique_ordered(
        [args.candidate_agent, *list(final_gate.get("learned_baseline_agents", []) or [])]
    )
    paper_main_rows = build_paper_main_comparison_rows(main_metric_rows, margin_rows, paper_table_agents)
    paper_paired_reward_rows = build_paper_paired_reward_rows(paired_rows)
    paper_support_reward_rows = build_paper_support_reward_rows(support_stat_rows)
    paper_prediction_setting_rows = build_paper_prediction_setting_rows(
        support_stat_rows,
        list(final_gate.get("prediction_required_settings", []) or []),
    )
    self_review_rows = build_self_review_rows(
        claim_checks=claim_checks,
        margin_rows=margin_rows,
        paired_rows=paired_rows,
        support_stat_rows=support_stat_rows,
        protocol_rows=protocol_rows,
        mappo_action_audit_rows=mappo_action_audit_rows,
        required_prediction_settings=list(final_gate.get("prediction_required_settings", []) or []),
    )
    self_review_blockers = [row for row in self_review_rows if row.get("status") == "block"]
    algorithm_comparison_rows = build_algorithm_comparison_rows(protocol_rows)
    audited_main_table_rows = build_audited_main_table_rows(
        main_metric_rows,
        protocol_rows,
        paired_rows,
    )
    training_fairness_rows = collect_training_fairness_rows(suite_reports, final_root)
    strongest_comparator_rows = build_strongest_comparator_audit_rows(
        main_metric_rows,
        list(final_gate.get("learned_baseline_agents", []) or []),
        list(final_gate.get("heuristic_reference_agents", []) or []),
    )
    reviewer_issue_rows = build_reviewer_issue_resolution_rows(
        final_gate,
        training_fairness_rows,
        strongest_comparator_rows,
        self_review_rows,
    )

    report = {
        "final_run_id": final_gate.get("run_id"),
        "final_gate_path": str(final_gate_path),
        "created_from": "build_top_journal_comparison_report.py",
        "paper_claim_ready": bool(final_gate.get("paper_claim_ready")),
        "review_ready": all(bool(check["passed"]) for check in claim_checks),
        "paper_ready_package_ready": all(bool(check["passed"]) for check in claim_checks)
        and not self_review_blockers,
        "learned_baseline_agents": list(final_gate.get("learned_baseline_agents", []) or []),
        "heuristic_reference_agents": list(final_gate.get("heuristic_reference_agents", []) or []),
        "paper_table_agents": paper_table_agents,
        "heuristic_policy": final_gate.get("heuristic_policy", "supplementary_reference_only"),
        "suite_report_paths": {suite_name: str(path) for suite_name, path, _ in suite_reports},
        "algorithm_comparison_table": algorithm_comparison_rows,
        "audited_main_table": audited_main_table_rows,
        "training_fairness_audit": training_fairness_rows,
        "strongest_comparator_audit": strongest_comparator_rows,
        "reviewer_issue_resolution": reviewer_issue_rows,
        "action_mix_summary_table": action_mix_summary_rows,
        "mappo_action_mix_audit": mappo_action_audit_rows,
        "claim_checks": claim_checks,
        "self_review_summary": {
            "blocker_count": len(self_review_blockers),
            "limitation_count": sum(1 for row in self_review_rows if row.get("status") == "limitation"),
            "pass_count": sum(1 for row in self_review_rows if row.get("status") == "pass"),
        },
        "claim_boundary": [
            "Primary claims are against the clean-retrained learned baselines listed in learned_baseline_agents.",
            "IPPO remains contract-blocked until a true independent per-agent wrapper is implemented.",
            "MAPPO is claimable only for artifacts that include mappo in learned_baseline_agents, pass duplicate-trace independence audits, and record the current controller head-credit v3 checkpoint protocol.",
            "If MAPPO underperforms PPO/DQN, the paper must include or cite the MAPPO action-mix audit and phrase the strongest learned-baseline claim against PPO.",
            "Controller-MAT is claimable only for artifacts that include controller_mat in learned_baseline_agents and pass duplicate-trace independence audits.",
            "DAG/cache/DT domain baselines are claimable only for artifacts that include dag_offload_drl, cache_offload_drl, and dt_handoff_drl in learned_baseline_agents and pass duplicate-trace independence audits.",
            "DDQN-family variants are optional only after passing duplicate-trace independence audits.",
            "reactive_greedy and popularity_cache_heuristic are supplementary non-learning references.",
            "Prediction setting-level claim checks are prediction-aware/calibrated feature-assistance checks unless a learned predictor checkpoint is attached; no_prediction and oracle_prediction remain diagnostic stress cases.",
            "Positive signed deltas favor SA-GHMAPPO; for lower-is-better metrics the raw delta is inverted in signed statistics.",
        ],
        "output_files": {
            "main_metric_table": str(output_dir / "main_metric_table.csv"),
            "reward_margins": str(output_dir / "reward_margins.csv"),
            "paired_statistics": str(output_dir / "paired_statistics.csv"),
            "support_metric_table": str(output_dir / "support_metric_table.csv"),
            "support_paired_statistics": str(output_dir / "support_paired_statistics.csv"),
            "baseline_protocol_matrix": str(output_dir / "baseline_protocol_matrix.csv"),
            "algorithm_comparison_table": str(output_dir / "algorithm_comparison_table.csv"),
            "action_mix_summary_table": str(output_dir / "action_mix_summary.csv"),
            "mappo_action_mix_audit": str(output_dir / "mappo_action_mix_audit.csv"),
            "markdown_report": str(output_dir / "top_journal_comparison_report.md"),
            "json_report": str(output_dir / "top_journal_comparison_report.json"),
            "paper_ready_main_comparison_csv": str(paper_output_dir / "paper_ready_main_comparison.csv"),
            "paper_ready_main_comparison_tex": str(paper_output_dir / "paper_ready_main_comparison.tex"),
            "paper_ready_audited_main_table_csv": str(
                paper_output_dir / "paper_ready_audited_main_table.csv"
            ),
            "paper_ready_audited_main_table_tex": str(
                paper_output_dir / "paper_ready_audited_main_table.tex"
            ),
            "paper_ready_audited_main_table_md": str(
                paper_output_dir / "paper_ready_audited_main_table.md"
            ),
            "paper_ready_training_fairness_audit_csv": str(
                paper_output_dir / "paper_ready_training_fairness_audit.csv"
            ),
            "paper_ready_training_fairness_audit_tex": str(
                paper_output_dir / "paper_ready_training_fairness_audit.tex"
            ),
            "paper_ready_training_fairness_audit_md": str(
                paper_output_dir / "paper_ready_training_fairness_audit.md"
            ),
            "paper_ready_strongest_comparator_audit_csv": str(
                paper_output_dir / "paper_ready_strongest_comparator_audit.csv"
            ),
            "paper_ready_strongest_comparator_audit_tex": str(
                paper_output_dir / "paper_ready_strongest_comparator_audit.tex"
            ),
            "paper_ready_strongest_comparator_audit_md": str(
                paper_output_dir / "paper_ready_strongest_comparator_audit.md"
            ),
            "paper_ready_reviewer_issue_resolution_csv": str(
                paper_output_dir / "paper_ready_reviewer_issue_resolution.csv"
            ),
            "paper_ready_reviewer_issue_resolution_tex": str(
                paper_output_dir / "paper_ready_reviewer_issue_resolution.tex"
            ),
            "paper_ready_reviewer_issue_resolution_md": str(
                paper_output_dir / "paper_ready_reviewer_issue_resolution.md"
            ),
            "paper_ready_algorithm_comparison_csv": str(
                paper_output_dir / "paper_ready_algorithm_comparison.csv"
            ),
            "paper_ready_algorithm_comparison_tex": str(
                paper_output_dir / "paper_ready_algorithm_comparison.tex"
            ),
            "paper_ready_paired_reward_statistics_csv": str(
                paper_output_dir / "paper_ready_paired_reward_statistics.csv"
            ),
            "paper_ready_paired_reward_statistics_tex": str(
                paper_output_dir / "paper_ready_paired_reward_statistics.tex"
            ),
            "paper_ready_support_reward_statistics_csv": str(
                paper_output_dir / "paper_ready_support_reward_statistics.csv"
            ),
            "paper_ready_support_reward_statistics_tex": str(
                paper_output_dir / "paper_ready_support_reward_statistics.tex"
            ),
            "paper_ready_prediction_setting_audit_csv": str(
                paper_output_dir / "paper_ready_prediction_setting_audit.csv"
            ),
            "paper_ready_prediction_setting_audit_tex": str(
                paper_output_dir / "paper_ready_prediction_setting_audit.tex"
            ),
            "paper_ready_mappo_action_mix_audit_csv": str(
                paper_output_dir / "paper_ready_mappo_action_mix_audit.csv"
            ),
            "paper_ready_mappo_action_mix_audit_tex": str(
                paper_output_dir / "paper_ready_mappo_action_mix_audit.tex"
            ),
            "paper_ready_self_review_csv": str(paper_output_dir / "paper_ready_self_review.csv"),
            "paper_ready_self_review_json": str(paper_output_dir / "paper_ready_self_review.json"),
            "paper_ready_report_md": str(paper_output_dir / "paper_ready_report.md"),
        },
    }

    write_csv(output_dir / "main_metric_table.csv", main_metric_rows)
    write_csv(output_dir / "reward_margins.csv", margin_rows)
    write_csv(output_dir / "paired_statistics.csv", paired_rows)
    write_csv(output_dir / "support_metric_table.csv", support_summary_rows)
    write_csv(output_dir / "support_paired_statistics.csv", support_stat_rows)
    write_csv(output_dir / "baseline_protocol_matrix.csv", protocol_rows)
    write_csv(output_dir / "algorithm_comparison_table.csv", algorithm_comparison_rows)
    write_csv(output_dir / "action_mix_summary.csv", action_mix_summary_rows)
    write_csv(output_dir / "mappo_action_mix_audit.csv", mappo_action_audit_rows)
    write_csv(paper_output_dir / "paper_ready_main_comparison.csv", paper_main_rows)
    write_csv(paper_output_dir / "paper_ready_audited_main_table.csv", audited_main_table_rows)
    write_csv(paper_output_dir / "paper_ready_training_fairness_audit.csv", training_fairness_rows)
    write_csv(paper_output_dir / "paper_ready_strongest_comparator_audit.csv", strongest_comparator_rows)
    write_csv(paper_output_dir / "paper_ready_reviewer_issue_resolution.csv", reviewer_issue_rows)
    write_csv(paper_output_dir / "paper_ready_algorithm_comparison.csv", algorithm_comparison_rows)
    write_csv(paper_output_dir / "paper_ready_paired_reward_statistics.csv", paper_paired_reward_rows)
    write_csv(paper_output_dir / "paper_ready_support_reward_statistics.csv", paper_support_reward_rows)
    write_csv(paper_output_dir / "paper_ready_prediction_setting_audit.csv", paper_prediction_setting_rows)
    write_csv(paper_output_dir / "paper_ready_mappo_action_mix_audit.csv", mappo_action_audit_rows)
    write_csv(paper_output_dir / "paper_ready_self_review.csv", self_review_rows)
    write_json(
        paper_output_dir / "paper_ready_self_review.json",
        {
            "final_run_id": final_gate.get("run_id"),
            "paper_ready_package_ready": report["paper_ready_package_ready"],
            "self_review_summary": report["self_review_summary"],
            "self_review_rows": self_review_rows,
        },
    )
    write_latex_table(
        paper_output_dir / "paper_ready_main_comparison.tex",
        caption=(
            "Main comparison on NGSIM + Alibaba. SA-GHMAPPO is compared against clean-retrained "
            "learned baselines; the popularity-cache result is reported only as a supplementary reference delta."
        ),
        label="tab:paper-ready-main-comparison",
        columns=[
            "Split",
            "Mode",
            "N",
            *[agent_label(agent_name) for agent_name in paper_table_agents],
            "Delta vs PPO",
            "Rel.",
            "Delta vs Pop.",
        ],
        rows=build_latex_main_rows(paper_main_rows, paper_table_agents),
        column_format="lll" + ("r" * (len(paper_table_agents) + 3)),
    )
    write_latex_table(
        paper_output_dir / "paper_ready_audited_main_table.tex",
        caption=(
            "Audited main comparison table. Rows are included only if they pass the current "
            "paper-ready audit as the main method, a primary learned comparator, or a supplementary "
            "non-learning reference. Delta CIs are SA-GHMAPPO minus the row algorithm for total reward."
        ),
        label="tab:paper-ready-audited-main-table",
        columns=[
            "Algorithm",
            "Role",
            "Audit",
            "Formal-M",
            "Formal-F",
            "Holdout-M",
            "Holdout-F",
            "Formal Delta CI",
            "Holdout Delta CI",
        ],
        rows=build_latex_audited_main_rows(audited_main_table_rows),
        column_format="lllrrrrll",
    )
    write_markdown_table(
        paper_output_dir / "paper_ready_audited_main_table.md",
        build_markdown_audited_main_rows(audited_main_table_rows),
        [
            "Algorithm",
            "Role",
            "Audit",
            "Formal Mixed",
            "Formal Full",
            "Holdout Mixed",
            "Holdout Full",
            "Formal CI",
            "Holdout CI",
            "Boundary",
        ],
    )
    write_markdown_table(
        paper_output_dir / "paper_ready_training_fairness_audit.md",
        training_fairness_rows,
        [
            "display_name",
            "audit_status",
            "records",
            "seeds",
            "episodes",
            "update_every",
            "update_count",
            "profile",
            "observation_contract",
            "action_contract",
            "trained_by_final_suite",
            "uses_sa_only_mechanisms",
            "fairness_basis",
        ],
    )
    write_markdown_table(
        paper_output_dir / "paper_ready_strongest_comparator_audit.md",
        strongest_comparator_rows,
        [
            "split",
            "window_protocol",
            "strongest_learned",
            "strongest_learned_reward",
            "ppo_rank_among_learned",
            "ppo_is_strongest_learned",
            "top3_learned_by_reward",
            "strongest_heuristic",
            "strongest_heuristic_reward",
            "decision_rule",
        ],
    )
    write_markdown_table(
        paper_output_dir / "paper_ready_reviewer_issue_resolution.md",
        reviewer_issue_rows,
        [
            "issue_id",
            "reviewer_concern",
            "status",
            "evidence",
            "paper_resolution",
            "remaining_action",
        ],
    )
    write_latex_table(
        paper_output_dir / "paper_ready_training_fairness_audit.tex",
        caption=(
            "Training-budget fairness audit for primary learned baselines. All rows use the "
            "matched NGSIM + Alibaba environment-interaction budget; algorithm-specific optimizers "
            "are preserved."
        ),
        label="tab:paper-ready-training-fairness",
        columns=["Algorithm", "Seeds", "Episodes", "Updates", "Obs.", "Action", "Status"],
        rows=build_latex_fairness_rows(training_fairness_rows),
        column_format="lllllll",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_strongest_comparator_audit.tex",
        caption=(
            "Strongest-comparator audit. The strongest learned baseline is selected by split-level "
            "reward ranking rather than assumed a priori."
        ),
        label="tab:paper-ready-strongest-comparator-audit",
        columns=["Split", "Mode", "Strongest Learned", "Reward", "PPO Rank", "Top-3", "Strongest Heuristic"],
        rows=build_latex_strongest_rows(strongest_comparator_rows),
        column_format="lllllll",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_reviewer_issue_resolution.tex",
        caption="Reviewer-concern resolution checklist for the main comparison package.",
        label="tab:paper-ready-reviewer-issue-resolution",
        columns=["ID", "Concern", "Status", "Resolution"],
        rows=build_latex_issue_rows(reviewer_issue_rows),
        column_format="llll",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_algorithm_comparison.tex",
        caption=(
            "Comparator protocol and claimability matrix. Claimable learned rows use the matched "
            "NGSIM + Alibaba environment-interaction budget; available rows require a new final-submission "
            "rerun, and optional DQN-family rows require duplicate-trace independence audit before citation."
        ),
        label="tab:paper-ready-algorithm-comparison",
        columns=["Comparator", "Role", "Family", "Contract", "Budget", "Status", "Boundary"],
        rows=build_latex_algorithm_rows(algorithm_comparison_rows),
        column_format="lllllll",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_paired_reward_statistics.tex",
        caption="Cluster-bootstrap paired total-reward statistics for SA-GHMAPPO against learned baselines.",
        label="tab:paper-ready-paired-reward",
        columns=["Split", "Baseline", "Pairs", "Clusters", "Mean Delta", "95% CI", "W/T/L", "p"],
        rows=build_latex_paired_rows(paper_paired_reward_rows),
        column_format="llrrrrrr",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_support_reward_statistics.tex",
        caption="Support-suite total-reward statistics for prediction robustness, system robustness, and scalability.",
        label="tab:paper-ready-support-reward",
        columns=["Suite", "Baseline", "Pairs", "Mean Delta", "95% CI", "Evidence"],
        rows=build_latex_support_rows(paper_support_reward_rows),
        column_format="llrrrl",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_prediction_setting_audit.tex",
        caption="Prediction-setting audit against PPO. Claim wording is limited to prediction-aware/calibrated feature assistance unless learned predictor checkpoints are attached.",
        label="tab:paper-ready-prediction-audit",
        columns=["Setting", "Scope", "Pairs", "Mean Delta", "95% CI", "Evidence"],
        rows=build_latex_prediction_rows(paper_prediction_setting_rows),
        column_format="llrrrl",
    )
    write_latex_table(
        paper_output_dir / "paper_ready_mappo_action_mix_audit.tex",
        caption=(
            "MAPPO action-mix audit. Negative reward and continuity deltas are reported as a "
            "reviewer-facing limitation of the controller-level MAPPO comparator, not as standalone "
            "evidence for the main method."
        ),
        label="tab:paper-ready-mappo-action-mix-audit",
        columns=[
            "Split",
            "Mode",
            "Reference",
            "Reward Delta",
            "Continuity Delta",
            "Failure Delta",
            "MAPPO Prefetch",
            "Ref Prefetch",
            "MAPPO Local",
            "Ref Local",
            "Risk",
        ],
        rows=build_latex_mappo_action_mix_rows(mappo_action_audit_rows),
        column_format="lll" + ("r" * 7) + "l",
    )
    paper_report = build_paper_result_summary(
        final_run_id=str(final_gate.get("run_id")),
        learned_baseline_agents=list(final_gate.get("learned_baseline_agents", []) or []),
        protocol_rows=protocol_rows,
        package_ready=bool(report["paper_ready_package_ready"]),
        paper_main_rows=paper_main_rows,
        paper_paired_rows=paper_paired_reward_rows,
        paper_support_rows=paper_support_reward_rows,
        prediction_rows=paper_prediction_setting_rows,
        mappo_action_audit_rows=mappo_action_audit_rows,
        self_review_rows=self_review_rows,
    )
    (paper_output_dir / "paper_ready_report.md").write_text(paper_report, encoding="utf-8")
    write_json(output_dir / "top_journal_comparison_report.json", report)
    markdown = build_markdown(
        report=report,
        margin_rows=margin_rows,
        paired_rows=paired_rows,
        support_stats=support_stat_rows,
        protocol_rows=protocol_rows,
        mappo_action_audit_rows=mappo_action_audit_rows,
    )
    (output_dir / "top_journal_comparison_report.md").write_text(markdown, encoding="utf-8")

    print("top-journal comparison report complete")
    print(f"review_ready: {str(report['review_ready']).lower()}")
    print(f"paper_ready_package_ready: {str(report['paper_ready_package_ready']).lower()}")
    print(f"report_dir: {output_dir}")
    print(f"paper_ready_dir: {paper_output_dir}")


def _run_cli() -> None:
    # Keep the script-path CLI on the same import path used by tests and runbooks.
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts import build_top_journal_comparison_report as module

    module.main()


if __name__ == "__main__":
    _run_cli()
