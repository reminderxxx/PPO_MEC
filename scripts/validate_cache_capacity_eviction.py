"""Validate optional cache capacity, LRU eviction, and telemetry export."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction, RSUState, WorkflowGraphState, WorkflowNode


OUTPUT_DIR = Path("artifacts/analysis/cache_capacity_eviction_telemetry_round9")
REQUIRED_TELEMETRY_FIELDS = [
    "cache_capacity_enabled",
    "cache_capacity_unit",
    "rsu_adapter_slots",
    "cache_capacity",
    "cache_used_size",
    "cache_remaining_size",
    "cache_occupancy_rate",
    "cache_admission_count",
    "cache_admission_added_new_adapter",
    "eviction_count",
    "evicted_adapter_count",
    "cache_hit",
    "cache_applied",
    "warm_hit",
    "cross_rsu_cold_start",
    "predictive_prefetch_requested",
    "prefetch_validated_hit",
    "migration_prepare_requested",
    "migration_prepare_realized",
    "backhaul_traffic_cost",
]


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


def _safe(value: Any) -> Any:
    if value is None or value == "":
        return "missing"
    return value


def _build_workflow() -> WorkflowGraphState:
    adapters = [
        "adapter_perception",
        "adapter_tracking",
        "adapter_fusion",
        "adapter_intent",
        "adapter_control",
    ]
    nodes = [
        WorkflowNode(
            node_id=f"node_{index + 1}",
            node_name=f"validation_node_{index + 1}",
            required_base_model="veh_base_v1",
            required_adapter=adapter_id,
            input_size=8,
            output_size=4,
            predecessors=[f"node_{index}"] if index > 0 else [],
            successors=[f"node_{index + 2}"] if index < len(adapters) - 1 else [],
        )
        for index, adapter_id in enumerate(adapters)
    ]
    edges = [(f"node_{index}", f"node_{index + 1}") for index in range(1, len(adapters))]
    return WorkflowGraphState(
        workflow_id="cache_capacity_validation_workflow",
        nodes=nodes,
        edges=edges,
        execution_order=[node.node_id for node in nodes],
        current_node_id=nodes[0].node_id,
    )


def _build_catalog() -> AdapterCatalog:
    adapters = [
        "adapter_perception",
        "adapter_tracking",
        "adapter_fusion",
        "adapter_intent",
        "adapter_control",
    ]
    return AdapterCatalog.from_dict(
        {
            "vehicle_base_models": [
                {
                    "base_model_id": "veh_base_v1",
                    "family": "validation_base_model",
                    "memory_mb": 768.0,
                }
            ],
            "rsu_adapter_caches": [{"rsu_id": "rsu_a", "cached_adapter_ids": []}],
            "adapter_state_bundles": [],
            "cache_objects": [
                {
                    "object_id": f"cache_obj_{adapter_id}",
                    "adapter_id": adapter_id,
                    "size_mb": 64.0,
                    "source": "validation_catalog",
                }
                for adapter_id in adapters
            ],
        }
    )


def _build_mobility() -> ReplayProvider:
    frames = [
        {
            "time_index": index,
            "vehicles": [
                {
                    "vehicle_id": "veh_1",
                    "position_x": 0.0,
                    "position_y": 0.0,
                    "speed": 0.0,
                    "base_model_id": "veh_base_v1",
                    "active_workflow_id": "cache_capacity_validation_workflow",
                }
            ],
        }
        for index in range(8)
    ]
    return ReplayProvider(trajectory_frames=frames)


def _build_control(adapter_id: str) -> ControlAction:
    return ControlAction(
        cache_action={
            "operation": "cache",
            "rsu_id": "rsu_a",
            "adapter_id": adapter_id,
            "strategy": "validation_cache_fill",
            "prediction_driven": False,
        },
        offload_action={
            "mode": "rsu",
            "target_rsu_id": "rsu_a",
            "strategy": "validation_current_rsu_offload",
        },
        migration_action={"mode": "keep", "strategy": "none"},
    )


def _run_case(case_name: str, cache_capacity_profile: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = VecWorkflowCoreEnv(
        mobility_provider=_build_mobility(),
        workflow_state=_build_workflow(),
        adapter_catalog=_build_catalog(),
        rsu_states=[RSUState(rsu_id="rsu_a", position_x=0.0, position_y=0.0, coverage_radius=100.0)],
        max_steps=8,
        cache_capacity_profile=cache_capacity_profile,
    )
    env.reset()
    event_rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    step_index = 0
    while not terminated and not truncated and step_index < 8:
        current_node = env.workflow_state.current_node()
        if current_node is None:
            break
        step_index += 1
        _, reward, terminated, truncated, info = env.step(_build_control(current_node.required_adapter))
        metrics = info.get("metrics_protocol", {})
        rsu_cache = next((rsu.cached_adapter_ids for rsu in env.rsu_states if rsu.rsu_id == "rsu_a"), [])
        event_rows.append(
            {
                "case_name": case_name,
                "step_index": step_index,
                "required_adapter": current_node.required_adapter,
                "reward_total": round(float(reward.total), 6),
                "rsu_cache_after_step": ";".join(rsu_cache) if rsu_cache else "empty",
                **{field: _safe(metrics.get(field)) for field in REQUIRED_TELEMETRY_FIELDS},
                "evicted_adapter_id": _safe(metrics.get("evicted_adapter_id")),
            }
        )
    eviction_count = sum(int(row["eviction_count"]) if str(row["eviction_count"]).isdigit() else 0 for row in event_rows)
    max_cache_used = max(
        [
            int(row["cache_used_size"])
            for row in event_rows
            if str(row["cache_used_size"]).isdigit()
        ]
        or [0]
    )
    occupancy_values = [
        float(row["cache_occupancy_rate"])
        for row in event_rows
        if str(row["cache_occupancy_rate"]) not in {"missing", ""}
    ]
    summary = {
        "case_name": case_name,
        "cache_capacity_enabled": bool(cache_capacity_profile.get("enabled", False)),
        "rsu_adapter_slots": cache_capacity_profile.get("rsu_adapter_slots", "missing"),
        "step_count": len(event_rows),
        "eviction_count": eviction_count,
        "max_cache_used_size": max_cache_used,
        "cache_occupancy_rate_max": max(occupancy_values) if occupancy_values else "missing",
        "old_append_only_behavior_preserved": bool(case_name == "capacity_disabled" and eviction_count == 0),
        "capacity_limit_respected": bool(
            case_name == "capacity_disabled"
            or (max_cache_used <= int(cache_capacity_profile.get("rsu_adapter_slots", 0) or 0))
        ),
        "admission_added_new_adapter_count": sum(
            1 for row in event_rows if str(row["cache_admission_added_new_adapter"]).lower() == "true"
        ),
    }
    return summary, event_rows


def main() -> None:
    disabled_summary, disabled_events = _run_case(
        "capacity_disabled",
        {
            "enabled": False,
            "unit": "adapter_slots",
            "rsu_adapter_slots": 2,
            "eviction_policy": "lru",
            "telemetry_enabled": True,
        },
    )
    enabled_summary, enabled_events = _run_case(
        "capacity_enabled_slots_2",
        {
            "enabled": True,
            "unit": "adapter_slots",
            "rsu_adapter_slots": 2,
            "eviction_policy": "lru",
            "telemetry_enabled": True,
        },
    )
    all_events = disabled_events + enabled_events
    coverage_rows = []
    for field in REQUIRED_TELEMETRY_FIELDS + ["evicted_adapter_id"]:
        present_count = sum(1 for row in all_events if row.get(field) not in (None, "", "missing"))
        coverage_rows.append(
            {
                "field_name": field,
                "present_row_count": present_count,
                "total_row_count": len(all_events),
                "coverage_rate": round(present_count / len(all_events), 6) if all_events else 0.0,
                "coverage_status": "observed" if present_count else "missing",
            }
        )
    diagnosis = {
        "task_name": "cache_capacity_eviction_telemetry_round9",
        "training_run": False,
        "reward_modified": False,
        "policy_modified": False,
        "baseline_modified": False,
        "capacity_disabled_preserves_old_behavior": bool(disabled_summary["old_append_only_behavior_preserved"]),
        "capacity_enabled_eviction_observed": bool(enabled_summary["eviction_count"] > 0),
        "capacity_enabled_limit_respected": bool(enabled_summary["capacity_limit_respected"]),
        "cache_occupancy_rate_observed": bool(enabled_summary["cache_occupancy_rate_max"] != "missing"),
        "summaries": [disabled_summary, enabled_summary],
        "generated_artifacts": {
            "cache_capacity_validation": str(OUTPUT_DIR / "cache_capacity_validation.csv"),
            "eviction_event_validation": str(OUTPUT_DIR / "eviction_event_validation.csv"),
            "telemetry_field_coverage": str(OUTPUT_DIR / "telemetry_field_coverage.csv"),
            "diagnosis_summary": str(OUTPUT_DIR / "diagnosis_summary.json"),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "cache_capacity_validation.csv", [disabled_summary, enabled_summary])
    _write_csv(OUTPUT_DIR / "eviction_event_validation.csv", all_events)
    _write_csv(OUTPUT_DIR / "telemetry_field_coverage.csv", coverage_rows)
    (OUTPUT_DIR / "diagnosis_summary.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print("cache capacity eviction validation complete")
    print(f"capacity_disabled_preserves_old_behavior: {diagnosis['capacity_disabled_preserves_old_behavior']}")
    print(f"capacity_enabled_eviction_observed: {diagnosis['capacity_enabled_eviction_observed']}")
    print(f"capacity_enabled_limit_respected: {diagnosis['capacity_enabled_limit_respected']}")
    print(f"cache_occupancy_rate_observed: {diagnosis['cache_occupancy_rate_observed']}")
    for key, path in diagnosis["generated_artifacts"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
