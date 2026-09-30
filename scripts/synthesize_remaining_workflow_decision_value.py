"""Synthesize separated native, local-measurement, and sensitivity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIB = 1024 * 1024


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_map(paired: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["instance_id"], row["rule_id"]): row
        for row in paired["episodes"]
    }


def synthesize(
    ledger: dict[str, Any],
    local: dict[str, Any],
    paired: dict[str, Any],
    reachability: dict[str, Any],
) -> dict[str, Any]:
    episodes = episode_map(paired)
    ablated_rule = "remaining_reuse_information_ablated"
    prepare_rule = "remaining_workflow_handoff_prepare"
    prefetch_rule = "remaining_workflow_reuse"
    long_instances = ("tail_length__long", "tail_adapter_structure__long")
    sensitivity_rows = []
    for instance in long_instances:
        ablated = episodes[(instance, ablated_rule)]["summary"]
        prepare = episodes[(instance, prepare_rule)]["summary"]
        prefetch = episodes[(instance, prefetch_rule)]["summary"]
        avoided_model_bytes = (
            ablated["transfer_bytes_by_type"].get("base_model", 0)
            + ablated["transfer_bytes_by_type"].get("adapter", 0)
            - prepare["transfer_bytes_by_type"].get("base_model", 0)
            - prepare["transfer_bytes_by_type"].get("adapter", 0)
        )
        prepare_realized = any(
            step["cache_event"].get("migration_realized") is True
            for step in episodes[(instance, prepare_rule)]["steps"]
        )
        native_state_bytes_observed = sum(
            int(round(float(step["cache_event"].get("state_migration_size_mb", 0.0)) * MIB))
            for step in episodes[(instance, prepare_rule)]["steps"]
        )
        state_availability = (
            "unavailable_prepare_realized_but_no_state_payload_bytes_emitted"
            if prepare_realized and native_state_bytes_observed == 0
            else "available_native_simulation_field"
        )
        sensitivity_rows.append(
            {
                "instance_id": instance,
                "equal_completion": prepare["node_completion_count"] == ablated["node_completion_count"],
                "equal_service_failures": prepare["service_failure_count"] == ablated["service_failure_count"],
                "avoided_native_model_transfer_bytes": avoided_model_bytes,
                "native_state_transfer_bytes_raw": native_state_bytes_observed,
                "native_state_transfer_availability": state_availability,
                "byte_break_even_condition": f"true_state_payload_bytes < {avoided_model_bytes}",
                "byte_break_even_mib": avoided_model_bytes / MIB,
                "measured_explicit_application_state_bytes": local["state_serialized_bytes"][0],
                "measured_state_is_complete_migration_payload": False,
                "narrow_application_state_net_bytes_if_used_as_an_explicit_assumption": avoided_model_bytes
                - local["state_serialized_bytes"][0],
                "network_time_formula_without_assumed_throughput": "delta_seconds=(true_state_payload_bytes-avoided_native_model_transfer_bytes)*8/link_bits_per_second; fixed protocol/queue latency unavailable",
                "prefetch_only_completed_nodes_delta": prefetch["node_completion_count"] - ablated["node_completion_count"],
                "prefetch_only_service_failures_delta": prefetch["service_failure_count"] - ablated["service_failure_count"],
                "interpretation": "prepare is byte-favorable in the native witness only if the unlogged true state payload and any fixed migration overhead stay below the stated break-even; the 732-byte application payload is not a complete runtime state and cannot close that condition",
            }
        )

    candidate_equal = ledger["same_completion_comparison"]["rows"]
    best = candidate_equal[0]
    worst = candidate_equal[-1]
    return {
        "artifact_run_id": "remaining_workflow_decision_value_audit_20260930_v1",
        "evidence_separation": {
            "native_simulation_ledger": "native_four_config_ledger.json and paired_witness_results.json",
            "local_host_measurement": "local_model_state_measurements.json",
            "assumption_sensitivity": "this file; algebraic thresholds only",
            "substitution_forbidden": True,
        },
        "native_four_config_key_result": {
            "equal_completion_scope": "candidate four configurations, 72 completed requests each",
            "minimum_transfer": best,
            "maximum_transfer": worst,
            "blocked_sharing_on_transfer_reduction_vs_interleaved_sharing_on_bytes": next(
                row["transfer_bytes_total"]
                for row in candidate_equal
                if row["config_id"] == "interleaved__sharing_on"
            )
            - next(
                row["transfer_bytes_total"]
                for row in candidate_equal
                if row["config_id"] == "blocked__sharing_on"
            ),
            "old_semantics_guard": "old lower transfer is inseparable from 36 or 60 failed services and is not an equal-completion saving",
        },
        "local_measurement_key_result": {
            "base_identity": local["weight_identity"],
            "process_first_load_median_ns": local["base_load_summary"]["median_ns"],
            "minimal_inference_median_ns": local["minimal_inference_summary"]["median_ns"],
            "minimal_inference_success_count": local["minimal_inference_success_count"],
            "adapter_switch_time": None,
            "adapter_status": local["adapter_inventory"]["status"],
            "explicit_application_state_bytes": local["state_serialized_bytes"][0],
            "state_save_median_ns": local["state_save_summary"]["median_ns"],
            "state_restore_median_ns": local["state_restore_summary"]["median_ns"],
            "state_restore_correct": local["state_restore_correct"],
            "guard": "local Apple M5 process-first load with uncontrolled OS page cache; not RSU/vehicle/network latency",
        },
        "action_reachability_key_result": {
            "reachable_migration_modes": reachability["reachable_migration_modes"],
            "future_adapter_direct_prepare": reachability["future_adapter_direct_prepare"],
            "explicit_state_migrate": reachability["explicit_state_migrate"],
            "continue_old_rsu_and_forward_result": reachability["continue_old_rsu_and_forward_result"],
        },
        "paired_witness_key_result": {
            "episode_count": paired["episode_count"],
            "decision_change_count": sum(row["decision_changed"] for row in paired["decision_changes"]),
            "two_step_matches_full_remaining_on_all_designed_instances": paired[
                "two_step_matches_full_remaining_on_all_designed_instances"
            ],
            "prepare_vs_ablated_long_instances": sensitivity_rows,
            "prefetch_only_guard": "action 1 reduced model transfer only by leaving one fewer node complete and adding one service failure at the fixed horizon; it is not a saving at equal service",
            "state_cost_guard": "action-4 prepare realized natively but emitted 0 state bytes; this is missing payload accounting, not a measured zero-cost migration",
        },
        "sensitivity_analysis": {
            "kind": "explicit algebraic assumptions",
            "rows": sensitivity_rows,
            "network_measurement": None,
            "link_rate_interval": None,
            "reason_no_interval": "no sourced deployment link interval was introduced; the byte break-even sign is reported symbolically for any positive link rate",
            "model_load_time_combination": "not performed because the local 1.015-GB SmolVLM base and the synthetic 96/128-MiB native cache objects are different identities",
        },
        "answers": {
            "remaining_workflow_changes_decision": "yes_in_two_long_tail_instances; no_in_two_short_or_nonreuse_instances",
            "change_improves_tradeoff": "bounded_yes_for_native_action4_model_bytes_at_equal_completion_and_failure_count; unverified_after_true_state_and_network_cost",
            "simple_lookahead_sufficient": "yes_for_all_four_pre_registered_instances; full remaining sequence never changed the initial action beyond two-step lookahead",
            "support_algorithm_change": "no; evidence favors exposing/accounting real state payload and action semantics before algorithm modification",
        },
        "problem_evidence_limit_next_step_table": [
            {
                "problem": "Does sharing/order change native cost at equal completion?",
                "evidence": "candidate four-way ledger, 72 successful requests per cell",
                "limit": "synthetic 96/128/8-MiB objects; no wait/execution/network timing",
                "next_step": "retain blocked+sharing-on as a mechanism witness; do not promote to deployment claim",
            },
            {
                "problem": "Are model and state costs real?",
                "evidence": "verified SmolVLM bytes/load/inference and 732-byte explicit application-state save/restore",
                "limit": "no compatible adapter; state is not full runtime/KV state; Apple M5 only",
                "next_step": "obtain one version-matched real adapter and an explicit complete state export/import contract",
            },
            {
                "problem": "Can remaining workflow information alter a legal decision?",
                "evidence": "6 initial-action changes across 16 bounded native episodes",
                "limit": "future-adapter direct prepare, explicit migrate, and old-RSU result forwarding are unreachable",
                "next_step": "freeze any expanded action contract before further performance comparison",
            },
            {
                "problem": "Does the change pay for itself?",
                "evidence": "action 4 avoided 104 MiB native model transfer in each long instance at equal completion/failures",
                "limit": "native state bytes missing; wireless time unmeasured; local model identity differs",
                "next_step": "measure complete state payload and one real transport path before claiming net latency/cost benefit",
            },
            {
                "problem": "Is more than two-step lookahead needed?",
                "evidence": "two-step and full-tail initial decisions match on all pre-registered instances",
                "limit": "only four deterministic instances and four-step horizon",
                "next_step": "do not add a stronger oracle/algorithm until a fixed case demonstrates a decision not representable by two-step lookahead",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = synthesize(
        read(args.ledger),
        read(args.local),
        read(args.paired),
        read(args.reachability),
    )
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["answers"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
