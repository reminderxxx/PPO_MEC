"""episode 级指标记录器。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.metrics.paper_metrics import PaperMetricSet
from src.metrics.reducers import 聚合奖励拆解, 求和


class EpisodeRecorder:
    """在 env-wrapper 边界统一记录 episode 数据。"""

    def __init__(self, prefetch_validation_window: int = 6) -> None:
        self._metric_set = PaperMetricSet()
        self._prefetch_validation_window = max(1, int(prefetch_validation_window))
        self._episode_index = -1
        self._run_metadata: dict[str, Any] = {}
        self._initial_state: dict[str, Any] | None = None
        self._step_records: list[dict[str, Any]] = []
        self._pending_prefetches: list[dict[str, Any]] = []
        self._episode_status: dict[str, Any] = {}

    def start_episode(self, run_metadata: dict[str, Any] | None = None) -> None:
        self._episode_index += 1
        self._run_metadata = deepcopy(run_metadata or {})
        self._initial_state = None
        self._step_records = []
        self._pending_prefetches = []
        self._episode_status = {
            "completed": False,
            "terminated": False,
            "truncated": False,
            "total_steps": 0,
        }

    def record_reset(self, state: dict[str, Any], info: dict[str, Any]) -> None:
        del info
        self._initial_state = deepcopy(state)

    def record_step(
        self,
        state: dict[str, Any],
        info: dict[str, Any],
        reward_dict: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        metrics_info = deepcopy(info.get("metrics_protocol", {}))
        metrics_info["time_index"] = int(state.get("time_index", 0))
        metrics_info["reward_dict"] = deepcopy(reward_dict)
        metrics_info["terminated"] = bool(terminated)
        metrics_info["truncated"] = bool(truncated)
        metrics_info["action_id"] = info.get("action_id")
        metrics_info["action_name"] = info.get("action_name")
        metrics_info["control_action"] = deepcopy(info.get("control_action", {}))
        metrics_info.setdefault("predictive_prefetch_correct", False)
        metrics_info.setdefault("predictive_prefetch_validated", False)
        metrics_info.setdefault("predictive_prefetch_pending", False)
        metrics_info.setdefault("predictive_prefetch_validation_state", "not_applicable")
        metrics_info.setdefault("prefetch_target_rsu_match", False)
        metrics_info.setdefault("prefetch_validated_hit", False)
        metrics_info.setdefault("prefetch_expired_miss", False)
        self._step_records.append(metrics_info)

        current_record_index = len(self._step_records) - 1
        self._validate_pending_prefetches(current_record_index)
        self._register_pending_prefetch(current_record_index)

        self._episode_status = {
            "completed": bool(terminated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_steps": len(self._step_records),
        }

    def build_summary(self) -> dict[str, Any]:
        reward_dicts = [record["reward_dict"] for record in self._step_records]
        reward_summary = 聚合奖励拆解(reward_dicts)
        system_metrics = self._metric_set.compute(self._step_records, self._episode_status)
        workflow_info = {}
        if self._initial_state is not None:
            workflow = self._initial_state.get("workflow", {})
            workflow_info = {
                "workflow_id": workflow.get("workflow_id"),
                "planned_nodes": len(workflow.get("execution_order", [])),
            }

        prefetch_validation_summary = self._build_prefetch_validation_summary()
        return {
            "run_info": {
                "episode_index": self._episode_index,
                "total_steps": len(self._step_records),
                **deepcopy(self._run_metadata),
            },
            "workflow_info": workflow_info,
            "episode_status": deepcopy(self._episode_status),
            "reward_breakdown": reward_summary,
            "system_metrics": system_metrics,
            "policy_trace_brief": self._build_policy_trace_brief(),
            "prefetch_summary": self._build_prefetch_summary(),
            "handoff_summary": self._build_handoff_summary(),
            "action_state_alignment_summary": self._build_action_state_alignment_summary(),
            "prefetch_validation_window": self._prefetch_validation_window,
            "pending_prefetch_count": prefetch_validation_summary["pending_prefetch_count"],
            "validated_predictive_prefetch_count": prefetch_validation_summary["validated_predictive_prefetch_count"],
            "prefetch_validation_summary": prefetch_validation_summary,
            "step_trace": deepcopy(self._step_records),
        }

    def export_summary(self, output_path: str | Path) -> dict[str, Any]:
        summary = self.build_summary()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _register_pending_prefetch(self, current_record_index: int) -> None:
        record = self._step_records[current_record_index]
        if not record.get("predictive_prefetch_requested", False):
            return
        self._pending_prefetches.append(
            {
                "source_record_index": current_record_index,
                "target_rsu_id": record.get("cache_target_rsu_id"),
                "deadline_record_index": current_record_index + self._prefetch_validation_window,
            }
        )

    def _validate_pending_prefetches(self, current_record_index: int) -> None:
        current_record = self._step_records[current_record_index]
        current_associated_rsu_id = current_record.get("post_action_associated_rsu_id")
        remaining_prefetches: list[dict[str, Any]] = []
        for pending in self._pending_prefetches:
            source_record_index = pending["source_record_index"]
            source_record = self._step_records[source_record_index]
            step_gap = current_record_index - source_record_index
            if step_gap < 1:
                remaining_prefetches.append(pending)
                continue
            if (
                current_associated_rsu_id is not None
                and current_associated_rsu_id == pending["target_rsu_id"]
                and step_gap <= self._prefetch_validation_window
            ):
                source_record["prefetch_target_rsu_match"] = True
                source_record["prefetch_validated_hit"] = True
                source_record["predictive_prefetch_correct"] = True
                source_record["predictive_prefetch_validated"] = True
                source_record["predictive_prefetch_pending"] = False
                source_record["predictive_prefetch_validation_state"] = "validated_hit"
                source_record["predictive_prefetch_validated_at_step"] = current_record_index + 1
                source_record["predictive_prefetch_validation_gap"] = step_gap
                continue
            if current_record_index >= pending["deadline_record_index"]:
                source_record["prefetch_expired_miss"] = True
                source_record["predictive_prefetch_correct"] = False
                source_record["predictive_prefetch_validated"] = True
                source_record["predictive_prefetch_pending"] = False
                source_record["predictive_prefetch_validation_state"] = "expired_miss"
                source_record["predictive_prefetch_validation_gap"] = step_gap
                continue
            remaining_prefetches.append(pending)
        self._pending_prefetches = remaining_prefetches

    def _build_policy_trace_brief(self) -> list[dict[str, Any]]:
        policy_trace: list[dict[str, Any]] = []
        for step_index, record in enumerate(self._step_records, start=1):
            policy_trace.append(
                {
                    "step_index": step_index,
                    "action_id": record.get("action_id"),
                    "action_name": record.get("action_name"),
                    "cache_strategy": record.get("cache_strategy"),
                    "pre_action_associated_rsu_id": record.get("pre_action_associated_rsu_id"),
                    "post_action_associated_rsu_id": record.get("post_action_associated_rsu_id"),
                    "decision_cache_target_rsu_id": record.get("decision_cache_target_rsu_id"),
                    "cache_target_rsu_id": record.get("cache_target_rsu_id"),
                    "cache_target_corrected_by_handoff": record.get("cache_target_corrected_by_handoff"),
                    "migration_mode": record.get("migration_mode"),
                    "migration_prepare_requested": record.get("migration_prepare_requested"),
                    "migration_prepare_realized": record.get("migration_prepare_realized"),
                    "handoff_event_count": record.get("handoff_event_count"),
                    "handoff_ready": record.get("handoff_ready"),
                    "predictive_prefetch_validation_state": record.get("predictive_prefetch_validation_state"),
                }
            )
        return policy_trace

    def _build_prefetch_summary(self) -> dict[str, Any]:
        prefetch_action_count = 求和(
            [1.0 for record in self._step_records if record.get("action_name") == "predictive_next_rsu_prefetch"]
        )
        true_predictive_prefetch_count = 求和(
            [1.0 for record in self._step_records if record.get("predictive_prefetch_requested")]
        )
        predictive_prefetch_correct_count = 求和(
            [1.0 for record in self._step_records if record.get("predictive_prefetch_correct")]
        )
        reactive_cache_fill_count = 求和(
            [1.0 for record in self._step_records if record.get("reactive_cache_fill")]
        )
        prefetch_target_rsu_match_count = 求和(
            [1.0 for record in self._step_records if record.get("prefetch_target_rsu_match")]
        )
        prefetch_validated_hit_count = 求和(
            [1.0 for record in self._step_records if record.get("prefetch_validated_hit")]
        )
        prefetch_expired_miss_count = 求和(
            [1.0 for record in self._step_records if record.get("prefetch_expired_miss")]
        )
        return {
            "prefetch_action_count": int(prefetch_action_count),
            "true_predictive_prefetch_count": int(true_predictive_prefetch_count),
            "predictive_prefetch_correct_count": int(predictive_prefetch_correct_count),
            "reactive_cache_fill_count": int(reactive_cache_fill_count),
            "prefetch_target_rsu_match_count": int(prefetch_target_rsu_match_count),
            "prefetch_validated_hit_count": int(prefetch_validated_hit_count),
            "prefetch_expired_miss_count": int(prefetch_expired_miss_count),
        }

    def _build_handoff_summary(self) -> dict[str, Any]:
        handoff_total_count = 求和([record.get("handoff_event_count", 0.0) for record in self._step_records])
        migration_during_handoff_count = 求和(
            [1.0 for record in self._step_records if record.get("migration_during_handoff")]
        )
        handoff_ready_count = 求和(
            [1.0 for record in self._step_records if record.get("handoff_ready")]
        )
        handoff_failure_count = 求和(
            [1.0 for record in self._step_records if record.get("handoff_failed")]
        )
        migration_prepare_count = 求和(
            [1.0 for record in self._step_records if record.get("migration_prepare_requested")]
        )
        return {
            "handoff_total_count": int(handoff_total_count),
            "migration_during_handoff_count": int(migration_during_handoff_count),
            "handoff_migration_action_count": int(migration_during_handoff_count),
            "handoff_ready_count": int(handoff_ready_count),
            "handoff_failure_count": int(handoff_failure_count),
            "migration_prepare_count": int(migration_prepare_count),
        }

    def _build_action_state_alignment_summary(self) -> dict[str, Any]:
        reactive_cache_fill_corrected_count = 求和(
            [1.0 for record in self._step_records if record.get("cache_target_corrected_by_handoff")]
        )
        cache_target_mismatch_count = 求和(
            [1.0 for record in self._step_records if record.get("cache_target_alignment_mismatch")]
        )
        reactive_cache_fill_mismatch_count = 求和(
            [
                1.0
                for record in self._step_records
                if record.get("action_name") == "current_rsu_cache_fill"
                and record.get("cache_target_rsu_id") != record.get("post_action_associated_rsu_id")
            ]
        )
        return {
            "reactive_cache_fill_auto_redirect_count": int(reactive_cache_fill_corrected_count),
            "cache_target_post_association_mismatch_count": int(cache_target_mismatch_count),
            "cache_target_current_association_mismatch_count": int(reactive_cache_fill_mismatch_count),
        }

    def _build_prefetch_validation_summary(self) -> dict[str, Any]:
        validated_predictive_prefetch_count = 求和(
            [1.0 for record in self._step_records if record.get("predictive_prefetch_correct")]
        )
        pending_prefetch_count = len(self._pending_prefetches)
        validated_request_count = 求和(
            [1.0 for record in self._step_records if record.get("predictive_prefetch_validated")]
        )
        prefetch_target_rsu_match_count = 求和(
            [1.0 for record in self._step_records if record.get("prefetch_target_rsu_match")]
        )
        prefetch_validated_hit_count = 求和(
            [1.0 for record in self._step_records if record.get("prefetch_validated_hit")]
        )
        prefetch_expired_miss_count = 求和(
            [1.0 for record in self._step_records if record.get("prefetch_expired_miss")]
        )
        return {
            "prefetch_validation_window": self._prefetch_validation_window,
            "pending_prefetch_count": int(pending_prefetch_count),
            "validated_predictive_prefetch_count": int(validated_predictive_prefetch_count),
            "validated_prefetch_request_count": int(validated_request_count),
            "prefetch_target_rsu_match_count": int(prefetch_target_rsu_match_count),
            "prefetch_validated_hit_count": int(prefetch_validated_hit_count),
            "prefetch_expired_miss_count": int(prefetch_expired_miss_count),
        }
