"""Evaluate direction-matched baseline agents through the shared real-sample path."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import checkpoint_required_agents, get_algo_spec, list_evaluable_agents
from src.evaluators.main_results_support import (
    build_selected_workflow_states,
    load_checkpoint_metadata,
    load_window_bundle,
    run_real_episode,
)
from src.runtime.typed_model_cache_runtime import (
    load_runtime_catalog,
    resolve_model_cache_runtime,
    validate_checkpoint_provenance,
)
from src.metrics.formal_nullable_metrics import (
    FORMAL_NULLABLE_METRIC_AGGREGATION_CONTRACT_VERSION,
    reduce_nullable_metric_rows,
    validate_end_to_end_delay_reason,
)
from src.runtime.formal_invalid_run_registry import (
    reject_permanently_invalid_formal_references,
)


EVALUABLE_BASELINES = list_evaluable_agents()
CHECKPOINT_REQUIRED = checkpoint_required_agents()
SUMMARY_METRICS = [
    "total_reward",
    "end_to_end_workflow_delay",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "predictive_prefetch_precision",
    "predictive_prefetch_request_count",
    "validated_predictive_prefetch_count",
    "migration_prepare_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
    "handoff_total_count",
    "mechanism_realization_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direction-matched baseline agents")
    parser.add_argument("--agent_name", choices=EVALUABLE_BASELINES, default="ppo")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--model_cache_runtime_config", default="")
    parser.add_argument("--expected_train_window_plan_identity_path", default="")
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="stable_first")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--classical_cache_slots", type=int, default=3)
    parser.add_argument("--reward_positive_offset", type=float, default=5.0)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "eval" / "algo_pool"))
    args = parser.parse_args()
    if args.agent_name in CHECKPOINT_REQUIRED and not args.checkpoint_path:
        parser.error(f"--checkpoint_path is required for agent_name={args.agent_name}")
    return args


def build_eval_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["system_metrics"]
    handoff = summary["handoff_summary"]
    prefetch = summary["prefetch_summary"]
    validation = summary["prefetch_validation_summary"]
    mechanism_realized = int(
        validation.get("validated_predictive_prefetch_count", 0) > 0
        or handoff.get("handoff_ready_count", 0) > 0
        or handoff.get("migration_during_handoff_count", 0) > 0
    )
    endpoint_audit = summary.get("formal_request_execution_audit") or {}
    delay_reason = endpoint_audit.get("end_to_end_workflow_delay_availability")
    if delay_reason is not None:
        validate_end_to_end_delay_reason(
            metrics["end_to_end_workflow_delay"], delay_reason
        )
    return {
        "agent_name": summary["run_info"].get("agent_name"),
        "eviction_policy": summary["run_info"].get("eviction_policy", "not_applicable"),
        "workflow_id": summary["run_info"].get("workflow_id"),
        "window_id": summary["run_info"].get("window_id"),
        "seed": summary["run_info"].get("seed"),
        "primary_vehicle_selection": summary["run_info"].get("primary_vehicle_selection", "stable_first"),
        "model_cache_profile": summary["run_info"].get("model_cache_profile"),
        "runtime_contract_sha256": summary["run_info"].get("runtime_contract_sha256"),
        "typed_catalog_fingerprint": summary["run_info"].get("typed_catalog_fingerprint"),
        "checkpoint_provenance_status": summary["run_info"].get(
            "checkpoint_provenance_status", "not_applicable"
        ),
        "episode_success": bool(summary.get("episode_success", False)),
        "total_reward": float(summary["reward_breakdown"]["total"]["sum"]),
        "end_to_end_workflow_delay": metrics["end_to_end_workflow_delay"],
        "end_to_end_workflow_delay_availability": delay_reason,
        "workflow_continuity_rate": metrics["workflow_continuity_rate"],
        "handoff_failure_rate": metrics["handoff_failure_rate"],
        "handoff_ready_ratio": metrics["handoff_ready_ratio"],
        "adapter_warm_hit_ratio": metrics["adapter_warm_hit_ratio"],
        "cross_rsu_cold_start_frequency": metrics["cross_rsu_cold_start_frequency"],
        "backhaul_traffic_cost": metrics["backhaul_traffic_cost"],
        "adapter_state_migration_overhead": metrics["adapter_state_migration_overhead"],
        "predictive_prefetch_precision": metrics["predictive_prefetch_precision"],
        "handoff_total_count": handoff["handoff_total_count"],
        "handoff_ready_count": handoff["handoff_ready_count"],
        "migration_prepare_count": handoff["migration_prepare_count"],
        "migration_during_handoff_count": handoff["migration_during_handoff_count"],
        "predictive_prefetch_request_count": prefetch["true_predictive_prefetch_count"],
        "validated_predictive_prefetch_count": validation["validated_predictive_prefetch_count"],
        "mechanism_realization_rate": float(mechanism_realized),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_means(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    means, _ = reduce_nullable_metric_rows(rows, SUMMARY_METRICS)
    return means


def metric_availability(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _, availability = reduce_nullable_metric_rows(rows, SUMMARY_METRICS)
    return availability


def main() -> None:
    args = parse_args()
    reject_permanently_invalid_formal_references(
        [args.output_root, args.checkpoint_path]
    )
    runtime_contract = resolve_model_cache_runtime(
        args.model_cache_runtime_config or None,
        root=ROOT_DIR,
    )
    adapter_catalog = load_runtime_catalog(runtime_contract, root=ROOT_DIR)
    run_id = datetime.now().strftime(f"{args.agent_name}_eval_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / args.agent_name / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    workflow_states = build_selected_workflow_states(
        workflow_csv_path=args.workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )
    mobility_bundle = load_window_bundle(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        random_seed=args.random_seed,
    )
    checkpoint_map = {args.agent_name: args.checkpoint_path}
    algo_spec = get_algo_spec(args.agent_name)
    cache_capacity_profile = dict(runtime_contract["cache_capacity_profile"])
    if not cache_capacity_profile.get("enabled"):
        cache_capacity_profile = None
    if algo_spec.get("required_eviction_policy"):
        if cache_capacity_profile is None:
            if args.classical_cache_slots <= 0:
                raise ValueError("classical_cache_slots must be positive")
            cache_capacity_profile = {
                "enabled": True,
                "unit": "adapter_slots",
                "rsu_adapter_slots": args.classical_cache_slots,
            }
        cache_capacity_profile["eviction_policy"] = algo_spec["required_eviction_policy"]
        cache_capacity_profile["eviction_policy_config"] = (
            {"aging_interval": 8, "aging_factor": 0.5}
            if args.agent_name == "reactive_aging_lfu"
            else {}
        )
        if args.agent_name == "reactive_random":
            cache_capacity_profile["eviction_policy_seed"] = args.random_seed
    checkpoint_gate = {"status": "not_applicable", "checkpoint_sha256": None}
    if args.checkpoint_path and runtime_contract["model_cache_profile"] == "typed_base_adapter_state_v1":
        checkpoint_metadata = load_checkpoint_metadata(args.checkpoint_path)
        provenance = checkpoint_metadata.get("typed_runtime_provenance") or {}
        expected_window_identity = provenance.get("train_window_plan_identity") or {}
        if args.expected_train_window_plan_identity_path:
            expected_window_identity = json.loads(
                Path(args.expected_train_window_plan_identity_path).read_text(encoding="utf-8-sig")
            )
        checkpoint_gate = validate_checkpoint_provenance(
            args.checkpoint_path,
            expected_agent_name=args.agent_name,
            expected_seed=args.random_seed,
            expected_runtime_contract=runtime_contract,
            expected_reward_positive_offset=args.reward_positive_offset,
            expected_window_plan_identity=expected_window_identity,
        )
        if checkpoint_gate["status"] != "compatible":
            raise ValueError(f"typed checkpoint provenance gate failed: {checkpoint_gate}")
    rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    for workflow_state in workflow_states:
        summary = run_real_episode(
            root_dir=ROOT_DIR,
            agent_name=args.agent_name,
            checkpoint_map=checkpoint_map,
            workflow_state=workflow_state,
            workflow_source_path=args.workflow_csv_path,
            mobility_bundle=mobility_bundle,
            seed=args.random_seed,
            max_steps=args.max_steps,
            mobility_source=args.mobility_source,
            primary_vehicle_selection=args.primary_vehicle_selection,
            reward_positive_offset=args.reward_positive_offset,
            cache_capacity_profile=cache_capacity_profile,
            adapter_catalog_override=adapter_catalog,
            model_cache_runtime_contract=runtime_contract,
            run_metadata={
                "script": "scripts/eval_algo_pool_real_sample.py",
                "run_id": run_id,
                "evaluation_agent": args.agent_name,
                "primary_vehicle_selection": args.primary_vehicle_selection,
                "checkpoint_provenance_status": checkpoint_gate["status"],
                "checkpoint_sha256": checkpoint_gate.get("checkpoint_sha256"),
            },
        )
        episode_summaries.append(summary)
        rows.append(build_eval_row(summary))

    eval_csv_path = output_root / "eval.csv"
    summary_path = output_root / "summary.json"
    write_csv(eval_csv_path, rows)
    mean_metrics, mean_metric_availability = reduce_nullable_metric_rows(
        rows, SUMMARY_METRICS
    )
    summary_payload = {
        "run_id": run_id,
        "agent_name": args.agent_name,
        "algo_spec": algo_spec,
        "cache_capacity_profile": cache_capacity_profile,
        "resolved_model_cache_runtime": runtime_contract,
        "runtime_contract_sha256": runtime_contract["runtime_contract_sha256"],
        "checkpoint_provenance_validation": checkpoint_gate,
        "checkpoint_path": args.checkpoint_path,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "checkpoint_metadata": (
            load_checkpoint_metadata(args.checkpoint_path)
            if args.checkpoint_path
            else {
                "checkpoint_path": "",
                "config_profile": "non_checkpoint_agent",
                "run_id": args.agent_name,
                "is_smoke_checkpoint": False,
            }
        ),
        "output_dir": str(output_root),
        "eval_csv_path": str(eval_csv_path),
        "summary_json_path": str(summary_path),
        "window_metadata": mobility_bundle.rsu_metadata,
        "formal_nullable_metric_aggregation_contract_version": (
            FORMAL_NULLABLE_METRIC_AGGREGATION_CONTRACT_VERSION
        ),
        "mean_metrics": mean_metrics,
        "mean_metric_availability": mean_metric_availability,
        "rows": rows,
        "episode_summaries": episode_summaries,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("algo pool evaluation complete")
    print(f"run_id: {run_id}")
    print(f"eval_csv_path: {eval_csv_path}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
