"""Create-only public-state analysis of the frozen CSCWD action branches."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cscwd_prepared_state_event_chain import _public_state_hash
from scripts.run_calibrated_workflow_event_aux_abstention_ab import (
    _load_candidate_inputs,
)
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    _network_seconds,
)

PLAN = ROOT / "docs/project/cscwd_state_condition_action_evidence_plan_20261010.md"
PLAN_SHA256 = "5ad92451b29ba2bb5204e8cb6597ac2c6b34f4268caf914da81222f3b02aa936"
SOURCE_MANIFEST_SHA256 = "859abe71b5465cceeb20787a429ec932274177311d181a724f47e404cc35c4b7"
PROTOCOL = ROOT / "configs/experiment/calibrated_workflow_event_aux_abstention_ab_v1.json"
CANDIDATE = ROOT / "artifacts/experiments/cscwd_event_aux_abstention_ab_20261010_v1"
CONTROL = ROOT / "artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _same_float(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _prepared_category(row: Any) -> str:
    if not isinstance(row, dict) or not bool(row.get("known", False)):
        return "unknown"
    if bool(row.get("valid", False)):
        return "valid"
    if bool(row.get("exists", False)):
        return "exists_but_stale"
    return "missing"


def _bundle_at_rsu(
    semantic: dict[str, Any],
    rsu_id: str | None,
) -> dict[str, Any]:
    context = dict(semantic.get("calibrated_context", {}) or {})
    required = [str(item) for item in context.get("required_bundle_ids", [])]
    catalog = dict(context.get("object_catalog", {}) or {})
    rsu = next(
        (
            row
            for row in semantic.get("rsus", [])
            if str(row.get("rsu_id")) == str(rsu_id)
        ),
        None,
    )
    if rsu_id is None or not isinstance(rsu, dict) or not required or not catalog:
        return {
            "status": "unknown",
            "missing_ids": [],
            "missing_resident_bytes": None,
            "missing_transfer_bytes": None,
            "missing_load_seconds": None,
        }
    residents = {str(item) for item in rsu.get("typed_resident_object_ids", [])}
    missing = [item for item in required if item not in residents]
    try:
        resident_bytes = sum(int(catalog[item]["resident_bytes"]) for item in missing)
        transfer_bytes = sum(int(catalog[item]["transfer_bytes"]) for item in missing)
        load_seconds = sum(float(catalog[item]["load_seconds"]) for item in missing)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("public object catalog is incomplete") from error
    return {
        "status": "ready" if not missing else "missing",
        "missing_ids": missing,
        "missing_resident_bytes": resident_bytes,
        "missing_transfer_bytes": transfer_bytes,
        "missing_load_seconds": load_seconds,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (
        int(_bool(left["workflow_completed"])),
        int(_bool(left["on_time_workflow_completed"])),
        -int(left["service_failures"]),
        -float(left["elapsed_seconds"]),
        -int(left["model_transfer_bytes"]),
        -int(left["state_transfer_bytes"]),
        -int(left["input_transfer_bytes"]),
        -float(left["recompute_seconds"]),
    )
    right_values = (
        int(_bool(right["workflow_completed"])),
        int(_bool(right["on_time_workflow_completed"])),
        -int(right["service_failures"]),
        -float(right["elapsed_seconds"]),
        -int(right["model_transfer_bytes"]),
        -int(right["state_transfer_bytes"]),
        -int(right["input_transfer_bytes"]),
        -float(right["recompute_seconds"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _service_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (
        int(_bool(left["workflow_completed"])),
        int(_bool(left["on_time_workflow_completed"])),
        -int(left["service_failures"]),
    )
    right_values = (
        int(_bool(right["workflow_completed"])),
        int(_bool(right["on_time_workflow_completed"])),
        -int(right["service_failures"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _source_category(
    state: dict[str, Any],
    branches: list[dict[str, Any]],
) -> tuple[str, str]:
    candidate_action = int(branches[0]["candidate_action"])
    factual = next(row for row in branches if int(row["branch_action"]) == candidate_action)
    alternatives = [row for row in branches if row is not factual]
    dominators = [row for row in alternatives if _dominates(row, factual)]
    if dominators:
        actions = sorted(int(row["branch_action"]) for row in dominators)
        return "strictly_dominated_avoidable", f"candidate_action_dominated_by={actions}"
    recovery = [
        row
        for row in alternatives
        if int(row["branch_action"]) in {0, 2}
        and _bool(row["first_service_completed"])
        and not _bool(factual["first_service_completed"])
    ]
    if recovery:
        return (
            "current_service_recovery_tradeoff",
            f"candidate_first_service_failed; recovery_actions={sorted(int(row['branch_action']) for row in recovery)}",
        )
    action4 = next(
        (row for row in branches if int(row["branch_action"]) == 4),
        None,
    )
    if action4 is not None and not _bool(action4["first_service_completed"]):
        improved = [
            row
            for row in branches
            if row is not action4 and _service_dominates(action4, row)
        ]
        if improved:
            return (
                "future_prepare_tradeoff",
                f"action4_first_service_failed_but_service_dominates={sorted(int(row['branch_action']) for row in improved)}",
            )
    if len(branches) >= 2 and any(
        not _dominates(left, right) and not _dominates(right, left)
        for left, right in itertools.combinations(branches, 2)
    ):
        return "resource_pareto_exchange", "at_least_one_pair_is_non_dominated"
    return "unknown", "insufficient_or_nonseparable_existing_branches"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-branch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_branch_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite existing output")
    if _sha(PLAN) != PLAN_SHA256:
        raise RuntimeError("frozen analysis plan drift")
    source_manifest = source / "analysis_manifest.json"
    if _sha(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source branch manifest identity drift")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("branch_count") != 26:
        raise RuntimeError("source branch artifact incomplete")
    for name, digest in manifest["files"].items():
        if _sha(source / name) != digest:
            raise RuntimeError(f"source artifact hash drift: {name}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config, splits, _, _ = _load_candidate_inputs(protocol)
    instances = {
        str(row["design_id"]): row
        for split in ("regression", "frozen_check")
        for row in splits[split]
    }
    summary_rows = _read_csv(source / "branch_summary_rows.csv")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        by_source[row["source_id"]].append(row)
    if len(by_source) != 11 or sum(map(len, by_source.values())) != 26:
        raise RuntimeError("frozen source/branch denominator drift")

    candidate_ledgers: dict[str, list[dict[str, str]]] = {}
    control_ledgers: dict[str, list[dict[str, str]]] = {}
    for view in ("selected", "update96"):
        candidate_ledgers[view] = _read_csv(
            CANDIDATE / f"candidate_{view}_behavior_ledger.csv"
        )
        control_ledgers[view] = _read_csv(CONTROL / f"new_{view}_behavior_ledger.csv")

    public_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    prefix_replay_steps = 0
    for source_id in sorted(by_source):
        group = sorted(by_source[source_id], key=lambda row: int(row["branch_action"]))
        reference = group[0]
        view = reference["view"]
        split = reference["split"]
        seed = int(reference["seed"])
        design_id = reference["design_id"]
        divergence = int(reference["first_divergence_step"])

        def select(rows: list[dict[str, str]]) -> list[dict[str, str]]:
            chosen = [
                row
                for row in rows
                if row["method"] == "sa_ghmappo"
                and row["checkpoint_view"] == view
                and row["split"] == split
                and int(row["seed"]) == seed
                and row["design_id"] == design_id
            ]
            chosen.sort(key=lambda row: int(row["step_index"]))
            return chosen

        candidate_rows = select(candidate_ledgers[view])
        control_rows = select(control_ledgers[view])
        env = CalibratedContinuousWorkflowEnv(config, instances[design_id])
        observation, info = env.reset()
        for index in range(divergence):
            candidate_action = int(candidate_rows[index]["executed_action"])
            control_action = int(control_rows[index]["executed_action"])
            if candidate_action != control_action or env.step_index != index:
                raise RuntimeError("common prefix action mismatch")
            observation, _, _, _, info = env.step(candidate_action)
            prefix_replay_steps += 1
            recorded_clock = candidate_rows[index]["clock_seconds_after"]
            if not _same_float(info["transition"]["clock_seconds_after"], recorded_clock):
                raise RuntimeError("common prefix clock mismatch")

        semantic = info["semantic_state"]
        public_hash = _public_state_hash(observation, info)
        first_hashes = set()
        enriched: list[dict[str, Any]] = []
        for row in group:
            branch_path = source / row["branch_path"]
            payload = json.loads(branch_path.read_text(encoding="utf-8"))
            first = payload["steps"][0]
            first_hashes.add(first["public_state_sha256"])
            if first["public_state_sha256"] != public_hash:
                raise RuntimeError("reconstructed public state hash mismatch")
            outcome = payload["outcome"]
            cache_events = first.get("cache_events", [])
            target_prepare_committed = any(
                bool(event.get("committed")) for event in cache_events
            )
            branch_row = {
                **row,
                "public_state_sha256": public_hash,
                "first_service_completed": bool(first["service_completed"]),
                "first_step_cost_seconds": float(first["step_cost_seconds"]),
                "first_model_transfer_bytes": int(first["model_transfer_bytes"]),
                "first_state_transfer_bytes": int(first["state_transfer_bytes"]),
                "first_input_transfer_bytes": int(first["input_transfer_bytes"]),
                "first_recompute_seconds": float(first["recompute_seconds"]),
                "first_target_prepare_committed": target_prepare_committed,
                "first_cache_reasons": json.dumps(
                    [str(event.get("reason", "")) for event in cache_events],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "first_state_transfer_status": str(
                    (first.get("state_transfer") or {}).get("status", "none")
                ),
                **outcome,
            }
            enriched.append(branch_row)
            branch_rows.append(branch_row)
        if len(first_hashes) != 1:
            raise RuntimeError("branch public state identity drift")

        vehicles = list(semantic.get("vehicles", []) or [])
        primary_id = semantic.get("primary_vehicle_id")
        vehicle = next(
            (
                row
                for row in vehicles
                if str(row.get("vehicle_id")) == str(primary_id)
            ),
            vehicles[0] if vehicles else {},
        )
        current_rsu = vehicle.get("associated_rsu_id")
        predictions = dict(semantic.get("predictions", {}) or {})
        target_rsu = predictions.get(
            "predicted_first_handoff_rsu_by_vehicle", {}
        ).get(str(vehicle.get("vehicle_id", "")))
        current_bundle = _bundle_at_rsu(semantic, current_rsu)
        target_bundle = _bundle_at_rsu(semantic, target_rsu)
        context = dict(semantic.get("calibrated_context", {}) or {})
        link = dict(context.get("link", {}) or {})
        prepared = dict(context.get("prepared_state_prefix", {}) or {})
        estimated_mbps = float(link["estimated_mbps"])
        fixed_seconds = float(link["fixed_seconds"])
        contact_seconds = float(context["contact_budget_seconds"])
        state_bytes = int(context["state_bytes"])
        input_bytes = int(context["input_bytes"])
        compute_seconds = float(context["compute_seconds"])
        fallback_seconds = float(config["vehicle"]["fallback_seconds"])
        failure_seconds = float(config["objective"]["failed_service_seconds"])
        restore_seconds = float(config["measured_time_seconds"]["state_restore_overhead"])
        current_prepare_seconds = (
            _network_seconds(
                int(current_bundle["missing_transfer_bytes"] or 0),
                estimated_mbps,
                fixed_seconds,
            )
            + float(current_bundle["missing_load_seconds"] or 0.0)
            if current_bundle["status"] != "unknown"
            else None
        )
        target_prepare_seconds = (
            _network_seconds(
                int(target_bundle["missing_transfer_bytes"] or 0),
                estimated_mbps,
                fixed_seconds,
            )
            + float(target_bundle["missing_load_seconds"] or 0.0)
            + _network_seconds(state_bytes, estimated_mbps, fixed_seconds)
            + restore_seconds
            if target_bundle["status"] != "unknown"
            else None
        )
        remaining_deadline = float(instances[design_id]["deadline_seconds"]) - float(
            env.clock_seconds
        )
        estimated_action0_seconds = (
            compute_seconds + float(current_prepare_seconds)
            if current_prepare_seconds is not None
            else None
        )
        estimated_action2_seconds = (
            fallback_seconds
            + compute_seconds
            + _network_seconds(input_bytes, estimated_mbps, fixed_seconds)
        )
        estimated_action4_seconds = (
            (compute_seconds if current_bundle["status"] == "ready" else failure_seconds)
            + (
                float(target_prepare_seconds)
                if target_prepare_seconds is not None
                and target_prepare_seconds <= contact_seconds
                else 0.0
            )
        )
        category, category_reason = _source_category(
            {}, enriched
        )
        causal = dict(predictions.get("causal_provenance", {}) or {})
        public_rows.append(
            {
                "source_id": source_id,
                "stratum": reference["stratum"],
                "view": view,
                "split": split,
                "seed": seed,
                "design_id": design_id,
                "first_divergence_step": divergence,
                "public_state_sha256": public_hash,
                "action_mask": json.dumps(info["action_mask"], separators=(",", ":")),
                "candidate_action": int(reference["candidate_action"]),
                "control_action": int(reference["control_action"]),
                "observed_branch_actions": json.dumps(
                    [int(row["branch_action"]) for row in enriched],
                    separators=(",", ":"),
                ),
                "current_rsu_id": current_rsu,
                "predicted_target_rsu_id": target_rsu,
                "current_bundle_status": current_bundle["status"],
                "current_missing_object_ids": json.dumps(current_bundle["missing_ids"], separators=(",", ":")),
                "current_missing_resident_bytes": current_bundle["missing_resident_bytes"],
                "current_missing_transfer_bytes": current_bundle["missing_transfer_bytes"],
                "target_bundle_status": target_bundle["status"],
                "target_missing_object_ids": json.dumps(target_bundle["missing_ids"], separators=(",", ":")),
                "target_missing_resident_bytes": target_bundle["missing_resident_bytes"],
                "target_missing_transfer_bytes": target_bundle["missing_transfer_bytes"],
                "state_bytes": state_bytes,
                "input_bytes": input_bytes,
                "compute_seconds": compute_seconds,
                "estimated_link_mbps": estimated_mbps,
                "estimated_link_fixed_seconds": fixed_seconds,
                "contact_budget_seconds": contact_seconds,
                "remaining_deadline_seconds": remaining_deadline,
                "current_prepared_prefix": _prepared_category(prepared.get("current")),
                "current_missing_completed_fraction": (prepared.get("current") or {}).get("missing_completed_fraction"),
                "target_prepared_prefix": _prepared_category(prepared.get("predicted_target")),
                "target_missing_completed_fraction": (prepared.get("predicted_target") or {}).get("missing_completed_fraction"),
                "prediction_confidence": causal.get("confidence"),
                "prediction_uncertainty": predictions.get("prediction_uncertainty_by_vehicle", {}).get(str(vehicle.get("vehicle_id", ""))),
                "predicted_sequence": json.dumps(predictions.get("next_rsu_sequence", {}).get(str(vehicle.get("vehicle_id", "")), []), separators=(",", ":")),
                "estimated_current_prepare_seconds": current_prepare_seconds,
                "estimated_target_prepare_plus_state_seconds": target_prepare_seconds,
                "estimated_target_prepare_contact_feasible": (
                    None if target_prepare_seconds is None else target_prepare_seconds <= contact_seconds
                ),
                "estimated_action0_first_step_seconds": estimated_action0_seconds,
                "estimated_action0_fits_remaining_deadline": (
                    None if estimated_action0_seconds is None else estimated_action0_seconds <= remaining_deadline
                ),
                "estimated_action2_first_step_seconds": estimated_action2_seconds,
                "estimated_action2_fits_remaining_deadline": estimated_action2_seconds <= remaining_deadline,
                "estimated_action4_first_step_seconds": estimated_action4_seconds,
                "estimated_action4_fits_remaining_deadline": estimated_action4_seconds <= remaining_deadline,
                "evidence_category": category,
                "category_reason": category_reason,
            }
        )

        for left, right in itertools.combinations(enriched, 2):
            pair_rows.append(
                {
                    "source_id": source_id,
                    "left_action": int(left["branch_action"]),
                    "right_action": int(right["branch_action"]),
                    "left_dominates_right": _dominates(left, right),
                    "right_dominates_left": _dominates(right, left),
                    "left_service_dominates_right": _service_dominates(left, right),
                    "right_service_dominates_left": _service_dominates(right, left),
                    "left_first_service_completed": _bool(left["first_service_completed"]),
                    "right_first_service_completed": _bool(right["first_service_completed"]),
                    "left_minus_right_elapsed_seconds": float(left["elapsed_seconds"]) - float(right["elapsed_seconds"]),
                    "left_minus_right_model_bytes": int(left["model_transfer_bytes"]) - int(right["model_transfer_bytes"]),
                    "left_minus_right_recompute_seconds": float(left["recompute_seconds"]) - float(right["recompute_seconds"]),
                }
            )

    counts = Counter(row["evidence_category"] for row in public_rows)
    action_first_service = Counter()
    action_observed = Counter()
    action_outcomes = defaultdict(Counter)
    for row in branch_rows:
        action = int(row["branch_action"])
        action_observed[action] += 1
        action_first_service[action] += int(_bool(row["first_service_completed"]))
        action_outcomes[action]["workflow_completed"] += int(_bool(row["workflow_completed"]))
        action_outcomes[action]["on_time"] += int(_bool(row["on_time_workflow_completed"]))
        action_outcomes[action]["service_failures"] += int(row["service_failures"])

    output.mkdir(parents=True)
    _write_csv(output / "public_state_rows.csv", public_rows)
    _write_csv(output / "branch_outcome_rows.csv", branch_rows)
    _write_csv(output / "pairwise_pareto_rows.csv", pair_rows)
    summary = {
        "schema_version": "cscwd_state_condition_action_evidence_v1",
        "source_states": len(public_rows),
        "existing_branches": len(branch_rows),
        "new_action_branches": 0,
        "new_branch_env_steps": 0,
        "prefix_replay_steps": prefix_replay_steps,
        "training_steps": 0,
        "category_counts": dict(sorted(counts.items())),
        "action_counts": {
            str(action): {
                "observed": action_observed[action],
                "first_service_completed": action_first_service[action],
                **action_outcomes[action],
            }
            for action in sorted(action_observed)
        },
        "current_bundle_counts": dict(
            sorted(Counter(row["current_bundle_status"] for row in public_rows).items())
        ),
        "target_bundle_counts": dict(
            sorted(Counter(row["target_bundle_status"] for row in public_rows).items())
        ),
        "prepared_prefix_counts": {
            "current": dict(sorted(Counter(row["current_prepared_prefix"] for row in public_rows).items())),
            "target": dict(sorted(Counter(row["target_prepared_prefix"] for row in public_rows).items())),
        },
        "decision": "pending_evidence_review_no_training_authorization",
    }
    _write_json(output / "summary.json", summary)
    files = (
        "public_state_rows.csv",
        "branch_outcome_rows.csv",
        "pairwise_pareto_rows.csv",
        "summary.json",
    )
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": "cscwd_state_condition_action_evidence_manifest_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "plan_sha256": PLAN_SHA256,
            "source_branch_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_states": len(public_rows),
            "existing_branches": len(branch_rows),
            "new_action_branches": 0,
            "new_branch_env_steps": 0,
            "prefix_replay_steps": prefix_replay_steps,
            "training_steps": 0,
            "formal_holdout_reads": 0,
            "files": {name: _sha(output / name) for name in files},
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "states": len(public_rows),
                "branches": len(branch_rows),
                "prefix_replay_steps": prefix_replay_steps,
                "category_counts": dict(counts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
