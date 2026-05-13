"""论文风格系统指标计算。"""

from __future__ import annotations

from typing import Any

from src.metrics.reducers import 安全比率, 求和


class PaperMetricSet:
    """根据 episode step records 计算系统指标。"""

    def compute(
        self,
        step_records: list[dict[str, Any]],
        episode_status: dict[str, Any],
    ) -> dict[str, float]:
        """输出统一、可序列化的系统指标字典。"""
        total_steps = len(step_records)
        handoff_events = 求和([record["handoff_event_count"] for record in step_records])
        continuity_success = 求和(
            [1.0 for record in step_records if not record["stall_occurred"]]
        )
        handoff_failures = 求和(
            [1.0 for record in step_records if record["handoff_failed"]]
        )
        handoff_ready = 求和(
            [1.0 for record in step_records if record["handoff_ready"]]
        )
        warm_hits = 求和([1.0 for record in step_records if record["warm_hit"]])
        cold_starts = 求和(
            [1.0 for record in step_records if record["cross_rsu_cold_start"]]
        )
        backhaul_cost = 求和(
            [record["backhaul_traffic_cost"] for record in step_records]
        )
        migration_overhead = 求和(
            [record["adapter_state_migration_overhead"] for record in step_records]
        )
        predictive_prefetch_requests = 求和(
            [1.0 for record in step_records if record["predictive_prefetch_requested"]]
        )
        predictive_prefetch_validated_requests = 求和(
            [1.0 for record in step_records if record["predictive_prefetch_validated"]]
        )
        predictive_prefetch_hits = 求和(
            [1.0 for record in step_records if record["predictive_prefetch_correct"]]
        )
        end_to_end_delay = 0.0
        if total_steps > 0:
            end_to_end_delay = float(step_records[-1]["time_index"] - step_records[0]["time_index"] + 1)

        precision_denominator = predictive_prefetch_validated_requests
        if precision_denominator <= 0.0:
            precision_denominator = predictive_prefetch_requests

        return {
            "end_to_end_workflow_delay": round(end_to_end_delay, 6),
            "workflow_continuity_rate": 安全比率(continuity_success, total_steps),
            "handoff_failure_rate": 安全比率(handoff_failures, handoff_events),
            "handoff_ready_ratio": 安全比率(handoff_ready, handoff_events),
            "adapter_warm_hit_ratio": 安全比率(warm_hits, total_steps),
            "cross_rsu_cold_start_frequency": 安全比率(cold_starts, total_steps),
            "backhaul_traffic_cost": round(backhaul_cost, 6),
            "adapter_state_migration_overhead": round(migration_overhead, 6),
            "predictive_prefetch_precision": 安全比率(
                predictive_prefetch_hits,
                precision_denominator,
            ),
            "episode_completion_rate": 1.0 if episode_status.get("completed", False) else 0.0,
        }
