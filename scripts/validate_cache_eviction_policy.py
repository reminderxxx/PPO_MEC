"""Produce auditable LRU contract and G03 behavior-parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.model_catalog.adapter_catalog import AdapterCatalog, RSUAdapterCacheProfile
from src.envs.core.cache_eviction import LRUEvictionPolicy
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import CacheObject, ControlAction, RSUState


RUN_ID = "cache_eviction_policy_lru_validation_20260817_v1"
OUTPUT_DIR = Path("artifacts/analysis") / RUN_ID


def _env(*, cached: list[str], capacity: float, sizes: dict[str, float], unit: str = "mb") -> VecWorkflowCoreEnv:
    catalog = AdapterCatalog(
        vehicle_base_models=[],
        rsu_adapter_caches=[RSUAdapterCacheProfile("rsu_a", list(cached))],
        adapter_state_bundles=[],
        cache_objects=[CacheObject(f"obj:{key}", key, value, "validation") for key, value in sizes.items()],
    )
    profile: dict[str, Any] = {"enabled": True, "unit": unit, "eviction_policy": "lru"}
    if unit == "mb":
        profile["capacity_mb"] = capacity
    else:
        profile["rsu_adapter_slots"] = int(capacity)
    env = VecWorkflowCoreEnv(
        adapter_catalog=catalog,
        rsu_states=[RSUState("rsu_a", 0.0, 0.0, 100.0, list(cached))],
        cache_capacity_profile=profile,
    )
    env.reset()
    return env


def _admit(env: VecWorkflowCoreEnv, adapter_id: str) -> dict[str, Any]:
    return env._apply_cache_action(
        ControlAction(cache_action={"operation": "cache", "rsu_id": "rsu_a", "adapter_id": adapter_id}),
        None,
        "validation_node",
        adapter_id,
    )


def main() -> None:
    rows: list[dict[str, Any]] = []
    cases = [
        ("slot_single", ["a", "b"], 2, {"a": 1.0, "b": 1.0, "c": 1.0}, "adapter_slots", "c", ["a"]),
        ("mb_single", ["a", "b"], 7, {"a": 3.0, "b": 2.0, "c": 4.0}, "mb", "c", ["a"]),
        ("mb_multi", ["a", "b", "c"], 10, {"a": 3.0, "b": 3.0, "c": 4.0, "d": 6.0}, "mb", "d", ["a", "b"]),
    ]
    for name, cached, capacity, sizes, unit, incoming, expected in cases:
        env = _env(cached=cached, capacity=capacity, sizes=sizes, unit=unit)
        before = list(env.rsu_states[0].cached_adapter_ids)
        result = _admit(env, incoming)
        rows.append(
            {
                "case": name,
                "before": before,
                "incoming": incoming,
                "expected_g03_victims": expected,
                "actual_victims": result["evicted_adapter_ids"],
                "victim_parity": result["evicted_adapter_ids"] == expected,
                "final_cache": list(env.rsu_states[0].cached_adapter_ids),
                "admission_added": result["added_new_adapter"],
                "capacity_used": result["cache_used_size"],
                "capacity_remaining": result["cache_remaining_size"],
                "eviction_plan": result["eviction_plan"],
            }
        )

    hit_env = _env(cached=["a", "b"], capacity=2, sizes={"a": 1.0, "b": 1.0, "c": 1.0}, unit="adapter_slots")
    hit_env._episode_steps = 1
    assert hit_env._check_rsu_has_required_adapter("rsu_a", "a")
    hit_result = _admit(hit_env, "c")

    initial = _env(cached=["a", "b", "c"], capacity=7, sizes={"a": 2.0, "b": 3.0, "c": 4.0})
    reset_state_before = initial.export_cache_eviction_policy_state()
    initial.reset()
    reset_state_after = initial.export_cache_eviction_policy_state()

    tie = LRUEvictionPolicy()
    tie.reset(rsu_id="rsu_a", initial_resident_ids=[], current_step=0)
    tie.on_admission(rsu_id="rsu_a", object_id="z", current_step=2)
    tie.on_admission(rsu_id="rsu_a", object_id="a", current_step=2)
    tie_plan = tie.plan_victims(
        rsu_id="rsu_a",
        resident_ids=["z", "a"],
        resident_sizes={"z": 1.0, "a": 1.0},
        required_free_capacity=1.0,
        protected_object_id=None,
        capacity_unit="adapter_slots",
        current_step=2,
    )

    disabled = VecWorkflowCoreEnv(cache_capacity_profile={"enabled": False, "eviction_policy": "lru"})
    disabled.reset()
    disabled_result = disabled._apply_cache_action(
        ControlAction(cache_action={"operation": "cache", "rsu_id": "rsu_a", "adapter_id": "adapter_lane"}),
        None,
        "node",
        "adapter_lane",
    )

    event_env = VecWorkflowCoreEnv(
        cache_capacity_profile={
            "enabled": True,
            "unit": "adapter_slots",
            "rsu_adapter_slots": 1,
            "eviction_policy": "lru",
        }
    )
    event_env.reset()
    _, event_reward, _, _, event_info = event_env.step(
        ControlAction(
            cache_action={
                "operation": "cache",
                "adapter_id": "adapter_lane",
                "strategy": "reactive_cache_fill",
            },
            offload_action={"mode": "rsu"},
        )
    )
    event = event_info["cache_event"]
    legacy = event_info["metrics_protocol"]
    event_parity = {
        "reward_total": event_reward.total,
        "reward_matches_g03": abs(event_reward.total - 2.75) <= 1.0e-9,
        "final_cache": list(event_env.rsu_states[0].cached_adapter_ids),
        "final_cache_matches_g03": event_env.rsu_states[0].cached_adapter_ids == ["adapter_lane"],
        "cache_event": event,
        "cache_event_matches_g03": bool(
            event["event_schema_version"] == "1.1.0"
            and event["eviction_policy"] == "lru"
            and event["eviction_count"] == 1
            and event["evicted_adapter_ids"] == ["adapter_perception"]
            and event["admission_added"] is True
            and event["cache_used_before"] == event["cache_used_after"] == 1
        ),
        "legacy_telemetry": {
            key: legacy.get(key)
            for key in (
                "cache_eviction",
                "eviction_count",
                "evicted_adapter_id",
                "cache_admission_added_new_adapter",
                "cache_hit",
                "cache_used_size",
                "cache_remaining_size",
            )
        },
        "legacy_telemetry_matches_g03": bool(
            legacy.get("cache_eviction") is True
            and legacy.get("eviction_count") == 1
            and legacy.get("evicted_adapter_id") == "adapter_perception"
            and legacy.get("cache_admission_added_new_adapter") is True
            and legacy.get("cache_used_size") == 1
            and legacy.get("cache_remaining_size") == 0.0
        ),
    }

    diagnosis = {
        "run_id": RUN_ID,
        "purpose": "G03 structural parity; not a performance experiment",
        "policy_identity": tie.export_state(),
        "all_victim_cases_match_g03": all(row["victim_parity"] for row in rows),
        "multi_victim_order_matches_g03": rows[-1]["actual_victims"] == ["a", "b"],
        "hit_recency_victim": hit_result["evicted_adapter_ids"],
        "hit_recency_expected": ["b"],
        "tie_break_victim": tie_plan.ordered_victim_ids,
        "tie_break_expected": ["a"],
        "initial_trim_final_cache": list(initial.rsu_states[0].cached_adapter_ids),
        "reset_isolation": reset_state_before == reset_state_after,
        "disabled_victim_planning_absent": disabled_result["eviction_plan"] is None,
        "disabled_admission_behavior_preserved": disabled_result["added_new_adapter"] is True,
        "event_parity": event_parity,
    }
    if not all(
        [
            diagnosis["all_victim_cases_match_g03"],
            diagnosis["multi_victim_order_matches_g03"],
            diagnosis["hit_recency_victim"] == diagnosis["hit_recency_expected"],
            diagnosis["tie_break_victim"] == diagnosis["tie_break_expected"],
            diagnosis["initial_trim_final_cache"] == ["b", "c"],
            diagnosis["reset_isolation"],
            diagnosis["disabled_victim_planning_absent"],
            diagnosis["disabled_admission_behavior_preserved"],
            event_parity["reward_matches_g03"],
            event_parity["final_cache_matches_g03"],
            event_parity["cache_event_matches_g03"],
            event_parity["legacy_telemetry_matches_g03"],
        ]
    ):
        raise AssertionError(f"LRU parity validation failed: {diagnosis}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "victim_plan_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "diagnosis_summary.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "exported_policy_state.json").write_text(
        json.dumps(initial.export_cache_eviction_policy_state(), indent=2), encoding="utf-8"
    )
    print(f"LRU eviction policy validation passed: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
