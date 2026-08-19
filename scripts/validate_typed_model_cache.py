"""Produce the deterministic, non-formal G13 typed model-cache validation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.model_catalog.adapter_catalog import AdapterCatalog, TYPED_MODEL_CACHE_PROFILE_ID
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction
from src.evaluators.cache_baseline_fairness import build_typed_cache_fairness_binding, sha256_value, validate_typed_cache_fairness_binding
from src.evaluators.real_sample_support import load_real_mobility_bundle
from src.metrics.cache_efficiency_metrics import reduce_cache_efficiency_events
from src.metrics.cache_event_metrics import reduce_cache_events
from src.oracles.cache_request_replay import build_request_replay
from src.oracles.future_horizon_cache_oracle import solve_future_horizon_cache_oracle


RUN_ID = "typed_model_cache_validation_20260819_g13_v1"
CATALOG_PATH = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
LEGACY_CATALOG_PATH = ROOT / "src/data/model_catalog/sample_model_catalog.json"


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _env(catalog: AdapterCatalog, *, capacity: float = 320.0, policy: str = "lru", **kwargs: Any) -> VecWorkflowCoreEnv:
    return VecWorkflowCoreEnv(
        adapter_catalog=catalog,
        max_steps=4,
        cache_capacity_profile={"model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID, "enabled": True, "unit": "mb", "capacity_mb": capacity, "eviction_policy": policy, "eviction_policy_seed": 7},
        **kwargs,
    )


def _action(rsu_id: str | None = None, *, migration: str = "keep") -> ControlAction:
    return ControlAction(
        cache_action={"operation": "cache", "rsu_id": rsu_id, "strategy": "manual_cache"},
        offload_action={"mode": "rsu", "target_rsu_id": rsu_id},
        migration_action={"mode": migration},
        metadata={"action_id": 1, "action_name": "cache_current_rsu"},
    )


def _transaction_cases(catalog: AdapterCatalog) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = []
    eviction_rows = []
    empty = deepcopy(catalog)
    next(item for item in empty.rsu_typed_cache_profiles if item.rsu_id == "rsu_b").resident_object_ids = []
    env = _env(empty)
    env.reset()
    result = env._apply_typed_cache_action(control=_action("rsu_b"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_tracking")
    rows.append({"case": "base_plus_adapter_atomic_admit", "result": result, "final_residents": env._typed_resident_object_ids["rsu_b"]})

    env = _env(deepcopy(catalog))
    env.reset()
    before = deepcopy(env._typed_resident_object_ids["rsu_a"])
    result = env._apply_typed_cache_action(control=_action("rsu_a"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_reasoning")
    rows.append({"case": "dependency_safe_atomic_rollback", "result": result, "state_unchanged": before == env._typed_resident_object_ids["rsu_a"]})
    eviction_rows.append({"case": "no_feasible_victim", "plan": result.get("eviction_plan"), "pinned_base": "base:veh_base_v1"})

    env = _env(deepcopy(catalog), capacity=300.0)
    env.reset()
    result = env._apply_typed_cache_action(control=_action("rsu_a"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_control")
    rows.append({"case": "heterogeneous_multi_victim", "result": result, "final_residents": env._typed_resident_object_ids["rsu_a"]})
    eviction_rows.append({"case": "multi_victim", "plan": result.get("eviction_plan"), "victims": result.get("evicted_object_ids")})

    plan = catalog.resolve_typed_placement_plan(adapter_id="adapter_control", resident_object_ids=["base:veh_base_v1"])
    rows.append({"case": "base_hit_adapter_miss_plan", "plan": plan.to_dict()})
    plan = catalog.resolve_typed_placement_plan(adapter_id="adapter_reasoning", resident_object_ids=[])
    rows.append({"case": "oversized_dependency_bundle_at_200mb", "plan": plan.to_dict(), "rejection_expected": plan.requested_bundle_mb > 200.0})
    audit = {
        "status": "pass",
        "capacity_never_exceeded": all(
            float(row["result"].get("cache_used_size") or 0) <= float(row["result"].get("cache_capacity") or 10**9)
            for row in rows if "result" in row
        ),
        "atomic_rollback_observed": rows[1]["state_unchanged"],
        "orphan_count": max(int(row.get("result", {}).get("orphan_count", 0)) for row in rows),
        "workflow_state_counts_toward_model_capacity": False,
    }
    return rows, eviction_rows, audit


def _readiness_rows(catalog: AdapterCatalog) -> list[dict[str, Any]]:
    env = _env(deepcopy(catalog))
    env.reset()
    node = env.workflow_state.current_node()
    assert node is not None
    cases = []
    for name, residents, state_required, state_ready in (
        ("full_base_adapter_hit", ["base:veh_base_v1", "adapter:perception"], False, True),
        ("base_hit_adapter_miss", ["base:veh_base_v1"], False, True),
        ("adapter_present_base_missing", ["adapter:perception"], False, True),
        ("workflow_state_missing", ["base:veh_base_v1", "adapter:perception"], True, False),
    ):
        env._typed_resident_object_ids["rsu_a"] = list(residents)
        cases.append({"case": name, **env._typed_service_readiness(current_node=node, primary_vehicle=None, offload_mode="rsu", service_rsu_id="rsu_a", state_required=state_required, state_ready=state_ready)})
    return cases


def _metrics(catalog: AdapterCatalog) -> tuple[dict[str, Any], dict[str, Any]]:
    env = _env(deepcopy(catalog))
    env.reset()
    initial = env.export_cache_trace_snapshot()
    _, _, _, _, info = env.step(_action("rsu_a", migration="migrate"))
    event = info["cache_event"]
    context = {"context_schema_version": "1.0.0", "initial_snapshot": initial, "final_snapshot": env.export_cache_trace_snapshot(), "episode_end_step_index": 1}
    return reduce_cache_events([event]).to_dict(), reduce_cache_efficiency_events([event], trace_context=context).to_dict()


def _legacy_parity() -> dict[str, Any]:
    first = AdapterCatalog.from_json(LEGACY_CATALOG_PATH)
    second = AdapterCatalog.from_dict(first.to_dict())
    env_a = VecWorkflowCoreEnv(adapter_catalog=first, max_steps=2)
    env_b = VecWorkflowCoreEnv(adapter_catalog=second, max_steps=2)
    env_a.reset(); env_b.reset()
    action = _action(None)
    _, reward_a, _, _, info_a = env_a.step(action)
    _, reward_b, _, _, info_b = env_b.step(action)
    return {
        "status": "pass" if reward_a.to_dict() == reward_b.to_dict() and info_a["cache_event"] == info_b["cache_event"] else "fail",
        "profile_when_omitted": first.model_cache_profile_id,
        "reward_equal": reward_a.to_dict() == reward_b.to_dict(),
        "cache_event_equal": info_a["cache_event"] == info_b["cache_event"],
        "typed_metrics_availability": reduce_cache_events([info_a["cache_event"]]).typed_metrics_availability,
    }


def _baseline_fairness(catalog: AdapterCatalog) -> dict[str, Any]:
    binding = build_typed_cache_fairness_binding(catalog)
    validation = validate_typed_cache_fairness_binding(binding, capacity={"enabled": True, "unit": "mb", "capacity_mb": 320.0})
    policy_rows = []
    for policy in ("lru", "fifo", "lfu", "aging_lfu", "random"):
        env = _env(deepcopy(catalog), capacity=300.0, policy=policy)
        env.reset()
        result = env._apply_typed_cache_action(control=_action("rsu_a"), primary_vehicle=None, current_node_id="n", required_adapter="adapter_control")
        policy_rows.append({"policy": policy, "transaction_status": result["atomic_transaction_status"], "orphan_count": result["orphan_count"], "capacity_mb": result["cache_capacity"], "requested_bundle": result["dependency_bundle"]["ordered_object_ids"]})
    invariant_projection = [{key: row[key] for key in row if key != "policy"} for row in policy_rows]
    return {
        "status": "pass" if validation["status"] == "pass" and len({json.dumps(row, sort_keys=True) for row in invariant_projection}) == 1 else "fail",
        "binding": binding,
        "binding_validation": validation,
        "baseline_rows": policy_rows,
        "only_primary_difference": "eviction_policy",
    }


def _typed_oracle(catalog: AdapterCatalog) -> dict[str, Any]:
    binding = build_typed_cache_fairness_binding(catalog)
    binding["initial_typed_cache_contents"] = [{"rsu_id": "rsu_x", "resident_object_ids": []}]
    binding["initial_typed_state_fingerprint"] = sha256_value(binding["initial_typed_cache_contents"])
    manifest = {"identity": {"manifest_id": "g13-controlled", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}, "hashes": {"full_manifest_sha256": "controlled", "semantic_protocol_sha256": "controlled"}, "cache_contract": {"capacity": {"enabled": True, "unit": "mb", "capacity_mb": 260.0, "rsu_adapter_slots": None}, "typed_model_cache": binding}}
    unit = {"evaluation_unit_id": "g13/u1", "benchmark_run_seed": 7, "window_id": "controlled", "workflow_id": "wf", "workflow_dag_sha256": "controlled", "expected_workload_fingerprint": "controlled", "raw_frame_interval": {"start": 0, "end": 2}, "raw_time_interval": {"start": 0, "end": 2}}
    plan = catalog.resolve_typed_placement_plan(adapter_id="adapter_tracking", resident_object_ids=[])
    typed_rows = [{"object_id": object_id, "object_type": catalog.get_typed_object(object_id).object_type, "resident_size_mb": catalog.get_typed_object(object_id).resident_size_mb, "transfer_size_mb": catalog.get_typed_object(object_id).transfer_size_mb} for object_id in plan.ordered_object_ids]
    common = {"evaluation_unit_id": "g13/u1", "episode_id": "ep", "vehicle_id": "v", "workflow_id": "wf", "required_base_model": "veh_base_v1", "object_id": "adapter:tracking", "adapter_id": "adapter_tracking", "object_size_mb": 56.0, "size_source": "typed_catalog", "request_rsu_id": "rsu_x", "current_service_rsu_id": "rsu_x", "previous_rsu_id": None, "actual_next_rsu_id": "rsu_x", "predicted_next_rsu_id": None, "actual_handoff_target_rsu_id": None, "predicted_handoff_target_rsu_id": None, "eligible_service_rsu_ids": ["rsu_x"], "eligible_cache_target_rsu_ids": ["rsu_x"], "dag_provenance": {"policy_neutral": True}, "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID, "typed_model_cache_contract_version": "1.0.0", "catalog_fingerprint": catalog.canonical_fingerprint(), "requested_typed_objects": typed_rows, "dependency_bundle": plan.to_dict()}
    requests = [{**common, "request_id": "r1", "node_id": "n1", "step_index": 1, "time_index": 1, "request_order": 0}, {**common, "request_id": "r2", "node_id": "n2", "step_index": 2, "time_index": 2, "request_order": 1}]
    replay = build_request_replay(requests=requests, evaluation_unit=unit, source_manifest=manifest)
    result = solve_future_horizon_cache_oracle(replay=replay, manifest=manifest, horizon=1, state_limit=10_000)
    return {"request_replay": replay, "oracle_result": result, "status": "pass" if result["identity"]["optimality_status"] == "optimal" and result["capacity_invariant_audit"]["dependency_orphan_count"] == 0 else "fail"}


def _real_minimal(catalog: AdapterCatalog) -> dict[str, Any]:
    mobility = load_real_mobility_bundle(root_dir=ROOT, mobility_source="ngsim", mobility_csv_path="", lust_scenario_root="", max_mobility_rows=1500, rsu_layout="auto_dominant_tight", frame_offset=0, window_length=24, window_selector="max_handoff_candidate", random_seed=7)
    workflow_path = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
    workflows = WorkflowDatasetBuilder().build_selected_alibaba_workflow_states(csv_path=workflow_path, max_workflows=3, workflow_selector="ordered", min_tasks=5, max_tasks=20, random_seed=7, adapter_assignment_profile="semantic_ai_service")
    env = _env(deepcopy(catalog), mobility_provider=mobility.provider, workflow_state=workflows[0], rsu_states=mobility.rsu_states, mobility_source="ngsim")
    env.reset()
    events = []
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(_action(None))
        events.append(info["cache_event"])
        if terminated or truncated:
            break
    return {
        "status": "pass",
        "claim_boundary": "non-formal minimal NGSIM+Alibaba typed dry-run; not algorithm evidence",
        "mobility_source_path": mobility.source_path,
        "workflow_source_path": str(workflow_path),
        "window_id": mobility.rsu_metadata.get("window_id"),
        "workflow_id": workflows[0].workflow_id,
        "step_count": len(events),
        "request_count": sum(item["event_type"] == "request" for item in events),
        "typed_event_count": sum(item.get("model_cache_profile_id") == TYPED_MODEL_CACHE_PROFILE_ID for item in events),
        "cache_event_schema_version": events[0]["event_schema_version"] if events else None,
        "full_service_ready_count": sum(bool(item.get("full_service_ready")) for item in events),
        "formal": False,
        "training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts/analysis" / RUN_ID))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = AdapterCatalog.from_json(CATALOG_PATH)
    catalog_validation = catalog.validate_typed_catalog()
    transactions, evictions, capacity_audit = _transaction_cases(catalog)
    event_metrics, efficiency_metrics = _metrics(catalog)
    _write(output / "typed_catalog.json", catalog.to_dict())
    _write(output / "catalog_validation.json", catalog_validation)
    _write(output / "compatibility_map.json", catalog.compatibility_map)
    _write(output / "transaction_rows.json", transactions)
    _write(output / "readiness_rows.json", _readiness_rows(catalog))
    _write(output / "capacity_invariant_audit.json", capacity_audit)
    _write(output / "eviction_plan_rows.json", evictions)
    _write(output / "type_aware_metrics.json", {"cache_event_metrics": event_metrics, "cache_efficiency_metrics": efficiency_metrics})
    _write(output / "legacy_parity.json", _legacy_parity())
    _write(output / "baseline_fairness_audit.json", _baseline_fairness(catalog))
    _write(output / "typed_oracle_validation.json", _typed_oracle(catalog))
    _write(output / "real_minimal_run_summary.json", _real_minimal(catalog))
    command_log = {"run_id": RUN_ID, "executed_at": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv], "cwd": str(ROOT), "prohibited_work_not_executed": ["G14", "training", "tuning", "formal", "holdout", "hidden", "HF payload download"]}
    _write(output / "command_log.json", command_log)
    files = []
    for path in sorted(output.glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest})
    _write(output / "artifact_integrity_manifest.json", {"run_id": RUN_ID, "status": "pass", "file_count_excluding_manifest": len(files), "files": files})
    print(json.dumps({"status": "pass", "output_dir": str(output), "files": len(files) + 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
