"""Outcome-blind dev selection, checkpoint freeze, and completeness gate."""

from __future__ import annotations

import argparse
from functools import cmp_to_key
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import validate_protocol_v1_1
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file
from src.runtime.portable_resource_identity import add_portable_resource_arguments
from src.runtime.formal_agent_order import (
    FormalAgentOrderError,
    reject_permanently_invalid_run_references,
    resolve_formal_agent_order,
)
from src.metrics.formal_nullable_metrics import nullable_finite_value
from src.runtime.formal_invalid_run_registry import (
    PermanentlyInvalidFormalReferenceError,
    reject_permanently_invalid_formal_references,
)


INVALID_G14C_V3_RUN_ROOT = Path(
    "/private/tmp/ppo_mec_g14c_v3_a7c9e8e/artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260820_203430_g14c_v3"
).resolve()
INVALID_G14C_V1_V2_RUN_ROOTS = (
    (
        ROOT
        / "artifacts/experiments/typed_model_cache_formal"
        / "typed_model_cache_formal_20260820_g14c_351fdb8_v1"
    ).resolve(),
    (
        ROOT
        / "artifacts/experiments/typed_model_cache_formal"
        / "typed_model_cache_formal_20260820_164251_g14c_v2"
    ).resolve(),
)
INVALID_G14C_V4_RUN_ROOTS = (
    (
        ROOT
        / "artifacts/experiments/typed_model_cache_formal"
        / "typed_model_cache_formal_20260824_110016_g14c_v4"
    ).resolve(),
    (
        ROOT
        / "artifacts/experiments/typed_model_cache_formal"
        / "typed_model_cache_formal_20260824_235839_g14c_v4"
    ).resolve(),
)
INVALID_G14C_V5_RUN_ROOT = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260825_111625_g14c_v5"
).resolve()
INVALID_G14C_V6_RUN_ROOT = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260825_135122_g14c_v6"
).resolve()
INVALID_G14C_V7_RUN_ROOT = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260826_233222_g14c_v7"
).resolve()
INVALID_G14C_V8_RUN_ROOT = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260828_101804_g14c_v8"
).resolve()
INVALID_G14C_V9_RUN_ROOT = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260830_113339_g14c_v9"
).resolve()
INVALID_FORMAL_RUN_ROOTS = (
    *INVALID_G14C_V1_V2_RUN_ROOTS,
    INVALID_G14C_V3_RUN_ROOT,
    *INVALID_G14C_V4_RUN_ROOTS,
    INVALID_G14C_V5_RUN_ROOT,
    INVALID_G14C_V6_RUN_ROOT,
    INVALID_G14C_V7_RUN_ROOT,
    INVALID_G14C_V8_RUN_ROOT,
    INVALID_G14C_V9_RUN_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=[
            "dev_select",
            "checkpoint_freeze",
            "integrity",
            "integrity_and_formal_gate",
            "formal_gate",
        ],
        required=True,
    )
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-path", required=True)
    add_portable_resource_arguments(parser)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


SELECTION_METRICS = (
    ("full_service_ready_byte_hit_rate", "maximize"),
    ("workflow_continuity_rate", "maximize"),
    ("transfer_mb_per_request", "minimize"),
    ("end_to_end_workflow_delay", "minimize"),
)


def _candidate_metric(row: dict[str, Any], field: str) -> float | None:
    if field not in row:
        raise ValueError(f"required dev selection metric missing: {field}")
    return nullable_finite_value(row[field], field=field)


def _compare_candidates(left: dict[str, Any], right: dict[str, Any]) -> int:
    for field, direction in SELECTION_METRICS:
        left_value = _candidate_metric(left, field)
        right_value = _candidate_metric(right, field)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return 1
        if right_value is None:
            return -1
        if left_value != right_value:
            if direction == "maximize":
                return -1 if left_value > right_value else 1
            return -1 if left_value < right_value else 1
    left_tie = (int(left["update_index"]), str(left["checkpoint_sha256"]))
    right_tie = (int(right["update_index"]), str(right["checkpoint_sha256"]))
    return (left_tie > right_tie) - (left_tie < right_tie)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def scientific_candidate_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"checkpoint_path", "artifact_location", "runtime_resolution"}
    }


def dev_select(input_root: Path, protocol: dict) -> dict:
    try:
        reject_permanently_invalid_formal_references([input_root])
    except PermanentlyInvalidFormalReferenceError as exc:
        raise ValueError(str(exc)) from exc
    candidates = read_json(input_root / "checkpoint_candidates.json")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("checkpoint_candidates.json must contain a non-empty list")
    non_formal_rehearsal = all(
        bool(row.get("non_formal_rehearsal"))
        for row in candidates
        if isinstance(row, dict)
    )
    selected = []
    order_audit = None
    if protocol["typed_model_cache_formal_protocol_version"] in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        try:
            order_audit = resolve_formal_agent_order(protocol=protocol)
            reject_permanently_invalid_run_references([input_root])
        except FormalAgentOrderError as exc:
            raise ValueError(str(exc)) from exc
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("checkpoint candidate must be an object")
        if protocol["typed_model_cache_formal_protocol_version"] == "2.3.0":
            companion = row.get("selection_metric_availability")
            if not isinstance(companion, dict):
                raise ValueError("Protocol v2.3 candidate lacks metric availability counts")
            for field, _ in SELECTION_METRICS:
                counts = companion.get(field)
                if not isinstance(counts, dict):
                    raise ValueError(f"candidate availability missing: {field}")
                available = int(counts.get("available_count", -1))
                unavailable = int(counts.get("unavailable_count", -1))
                total = int(counts.get("total_count", -1))
                if (
                    total < 0
                    or available < 0
                    or unavailable < 0
                    or available + unavailable != total
                    or (_candidate_metric(row, field) is None) != (available == 0)
                ):
                    raise ValueError(f"candidate availability/value mismatch: {field}")
        key = (str(row["agent_name"]), int(row["seed"]), str(row["capacity_label"]))
        groups.setdefault(key, []).append(row)
    if order_audit is None:
        ordered_group_keys = sorted(groups)
    else:
        matrix = protocol["execution_contract"]["command_templates"]["train"][
            "matrix_contexts"
        ]
        capacities = (
            list(dict.fromkeys(str(row["capacity_label"]) for row in candidates))
            if non_formal_rehearsal
            else list(dict.fromkeys(str(row["capacity_label"]) for row in matrix))
        )
        seeds = (
            sorted({int(row["seed"]) for row in candidates})
            if non_formal_rehearsal
            else [int(seed) for seed in protocol["seed_plan"]["seeds"]]
        )
        selected_agents = (
            [
                agent
                for agent in order_audit["learned_agent_order"]
                if agent in {key[0] for key in groups}
            ]
            if non_formal_rehearsal
            else order_audit["learned_agent_order"]
        )
        if not selected_agents:
            raise ValueError("non-formal dev selection has no learned agent")
        ordered_group_keys = [
            (agent, seed, capacity)
            for capacity in capacities
            for agent in selected_agents
            for seed in seeds
        ]
        if set(groups) != set(ordered_group_keys):
            raise ValueError("dev checkpoint candidate group membership drift")
        order_hashes = {
            row.get("formal_agent_order_contract_semantic_sha256") for row in candidates
        }
        if order_hashes != {order_audit["semantic_sha256"]}:
            raise ValueError("dev checkpoint candidate order-contract identity drift")
    for key in ordered_group_keys:
        rows = groups[key]
        ranked = sorted(rows, key=cmp_to_key(_compare_candidates))
        winner = dict(ranked[0])
        selected.append(winner)
    training_identity_fields = (
        "agent_scientific_config_semantic_sha256",
        "formal_training_execution_binding_sha256",
        "formal_protocol_semantic_sha256",
        "execution_commit",
        "resolved_execution_context_sha256",
    )
    if protocol["typed_model_cache_formal_protocol_version"] in {"1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        training_identity_fields = (*training_identity_fields, "active_formal_bundle_sha256")
    if protocol["typed_model_cache_formal_protocol_version"] == "2.3.0":
        training_identity_fields = (
            *training_identity_fields,
            "formal_nullable_metric_aggregation_contract_semantic_sha256",
        )
    training_identity = None
    if (
        not non_formal_rehearsal
        and protocol["typed_model_cache_formal_protocol_version"] in {"1.6.0", "1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}
    ):
        identities = {
            tuple(row.get(field) for field in training_identity_fields)
            for row in candidates
        }
        if len(identities) != 1 or any(value in {None, ""} for value in next(iter(identities))):
            raise ValueError("dev candidates do not share one complete training identity")
        training_identity = dict(zip(training_identity_fields, next(iter(identities))))
        if training_identity["formal_protocol_semantic_sha256"] != protocol["hashes"]["semantic_sha256"]:
            raise ValueError("dev candidate active Protocol identity mismatch")
    metric_candidate_availability = {
        field: {
            "candidate_count": len(candidates),
            "available_candidate_count": sum(
                _candidate_metric(row, field) is not None for row in candidates
            ),
            "unavailable_candidate_count": sum(
                _candidate_metric(row, field) is None for row in candidates
            ),
            "dimension_participated": any(
                _candidate_metric(row, field) is not None for row in candidates
            ),
        }
        for field, _ in SELECTION_METRICS
    }
    return {
        "dev_checkpoint_selection_version": "1.1.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_split": "dev",
        "formal_or_holdout_used": False,
        "selection_rule": protocol["training_budget"]["checkpoint_selection"]["metric_rule"],
        "nullable_ordering_rule": (
            "finite_before_unavailable_per_dimension; both_unavailable_skips_dimension; "
            "no_numeric_sentinel"
        ),
        "selection_metric_candidate_availability": metric_candidate_availability,
        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol.get(
            "formal_nullable_metric_aggregation_contract", {}
        ).get("semantic_sha256"),
        "selected": selected,
        "selection_sha256": canonical_sha256(
            [scientific_candidate_projection(row) for row in selected]
        ),
        "checkpoint_locations_nonsemantic": True,
        "formal_training_identity": training_identity,
        "formal_agent_order_contract_semantic_sha256": (
            order_audit["semantic_sha256"] if order_audit else None
        ),
        "selected_agent_order": (
            list(dict.fromkeys(str(row["agent_name"]) for row in selected))
            if order_audit
            else None
        ),
        "non_formal_rehearsal": non_formal_rehearsal,
    }


def checkpoint_freeze(input_root: Path, protocol: dict) -> dict:
    try:
        reject_permanently_invalid_formal_references([input_root])
    except PermanentlyInvalidFormalReferenceError as exc:
        raise ValueError(str(exc)) from exc
    selection = read_json(input_root / "dev_selection.json")
    if selection.get("protocol_semantic_sha256") != protocol["hashes"]["semantic_sha256"]:
        raise ValueError("dev selection protocol hash mismatch")
    formal_training_identity = selection.get("formal_training_identity")
    order_audit = None
    if protocol["typed_model_cache_formal_protocol_version"] in {"1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}:
        try:
            order_audit = resolve_formal_agent_order(protocol=protocol)
        except FormalAgentOrderError as exc:
            raise ValueError(str(exc)) from exc
        if selection.get("formal_agent_order_contract_semantic_sha256") != order_audit[
            "semantic_sha256"
        ]:
            raise ValueError("dev selection formal agent order contract mismatch")
        observed_selected_agents = list(
            dict.fromkeys(str(row.get("agent_name")) for row in selection.get("selected", []))
        )
        expected_selected_agents = (
            [
                agent
                for agent in order_audit["learned_agent_order"]
                if agent in set(observed_selected_agents)
            ]
            if bool(selection.get("non_formal_rehearsal"))
            else order_audit["learned_agent_order"]
        )
        if observed_selected_agents != expected_selected_agents:
            raise ValueError("dev selection learned-agent order drift")
        if protocol["typed_model_cache_formal_protocol_version"] == "2.3.0" and selection.get(
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ) != protocol["formal_nullable_metric_aggregation_contract"]["semantic_sha256"]:
            raise ValueError("dev selection nullable metric contract mismatch")
    if (
        not bool(selection.get("non_formal_rehearsal"))
        and protocol["typed_model_cache_formal_protocol_version"] in {"1.6.0", "1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"}
    ):
        if not isinstance(formal_training_identity, dict):
            raise ValueError("dev selection lacks formal training identity")
        if formal_training_identity.get("formal_protocol_semantic_sha256") != protocol["hashes"]["semantic_sha256"]:
            raise ValueError("dev selection formal training Protocol identity mismatch")
    frozen = []
    for row in selection.get("selected", []):
        path = Path(row["checkpoint_path"]).resolve()
        try:
            reject_permanently_invalid_formal_references([path, row])
        except PermanentlyInvalidFormalReferenceError as exc:
            raise ValueError(
                "invalid G14C v3/v4 checkpoint reference rejected; "
                "the hard rejection set covers all G14C v1-v5 invalid runs "
                "and every later permanently invalid run through G14C v12; "
                + str(exc)
            ) from exc
        if any(path_is_within(path, root) for root in INVALID_FORMAL_RUN_ROOTS):
            raise ValueError(
                "invalid G14C v3/v4 checkpoint reference rejected; "
                "the hard rejection set covers all G14C v1-v5 invalid runs "
                "and G14C v6/v7"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != row.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint hash mismatch: {path}")
        typed = row.get("typed_runtime_provenance") or {}
        if formal_training_identity is not None:
            for field, expected in formal_training_identity.items():
                if row.get(field) != expected:
                    raise ValueError(f"selected checkpoint training identity mismatch: {field}")
        identity = {
            "checkpoint_sha256": digest,
            "agent": str(row["agent_name"]),
            "seed": int(row["seed"]),
            "capacity": str(row["capacity_label"]),
            "execution_commit": typed.get("execution_git_commit"),
            "training_protocol": protocol["hashes"]["semantic_sha256"],
            "catalog_identity": typed.get("typed_catalog_fingerprint"),
            "runtime_identity": row.get("runtime_contract_sha256"),
            "split_identity": protocol["identity"]["split_semantic_sha256"],
            "window_identity": typed.get("train_window_plan_identity"),
            "selection_metric_values": {
                field: row[field]
                for field in (
                    "full_service_ready_byte_hit_rate",
                    "workflow_continuity_rate",
                    "transfer_mb_per_request",
                    "end_to_end_workflow_delay",
                )
            },
            "selection_metric_availability": row.get(
                "selection_metric_availability", {}
            ),
            "formal_nullable_metric_aggregation_contract_semantic_sha256": row.get(
                "formal_nullable_metric_aggregation_contract_semantic_sha256"
            ),
            "update_index": int(row["update_index"]),
            "agent_scientific_config_semantic_sha256": row.get(
                "agent_scientific_config_semantic_sha256"
            ),
            "formal_training_execution_binding_sha256": row.get(
                "formal_training_execution_binding_sha256"
            ),
            "resolved_execution_context_sha256": row.get(
                "resolved_execution_context_sha256"
            ),
            "formal_agent_order_contract_semantic_sha256": row.get(
                "formal_agent_order_contract_semantic_sha256"
            ),
            "active_formal_bundle_sha256": row.get(
                "active_formal_bundle_sha256"
            ),
        }
        identity["semantic_identity_fingerprint"] = canonical_sha256(identity)
        frozen.append(
            {
                **row,
                "checkpoint_path": str(path),
                "checkpoint_identity": identity,
                "artifact_location": {
                    "original_location": str(path),
                    "resolved_location": str(path),
                    "location_is_scientific_identity": False,
                    "path_relocation_allowed_after_hash_validation": True,
                },
            }
        )
    return {
        "checkpoint_freeze_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "frozen_checkpoint_count": len(frozen),
        "frozen_checkpoints": frozen,
        "formal_or_holdout_used": False,
        "freeze_sha256": canonical_sha256(
            [row["checkpoint_identity"] for row in frozen]
        ),
        "checkpoint_location_contract_version": "1.0.0",
        "invalid_run_roots": [str(root) for root in INVALID_FORMAL_RUN_ROOTS],
        "formal_training_identity": formal_training_identity,
        "formal_agent_order_contract_semantic_sha256": (
            order_audit["semantic_sha256"] if order_audit else None
        ),
        "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol.get(
            "formal_nullable_metric_aggregation_contract", {}
        ).get("semantic_sha256"),
        "frozen_agent_order": order_audit["learned_agent_order"] if order_audit else None,
    }


def write_checkpoint_companions(input_root: Path, freeze: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in freeze["frozen_checkpoints"]:
        grouped.setdefault(str(row["capacity_label"]), []).append(row)
    outputs: list[dict] = []
    for capacity_label, rows in sorted(grouped.items()):
        seed_manifest: dict[str, dict[str, str]] = {}
        provenance_manifest: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            agent = str(row["agent_name"])
            seed = str(int(row["seed"]))
            seed_manifest.setdefault(agent, {})[seed] = str(row["checkpoint_path"])
            runtime_provenance = row.get("typed_runtime_provenance") or {}
            provenance_manifest.setdefault(agent, {})[seed] = {
                "checkpoint_sha256": row["checkpoint_sha256"],
                "execution_git_commit": runtime_provenance.get("execution_git_commit"),
                "train_window_plan_identity": runtime_provenance.get(
                    "train_window_plan_identity"
                ),
                "formal_protocol_semantic_sha256": freeze["protocol_semantic_sha256"],
                "agent_scientific_config_semantic_sha256": row.get(
                    "agent_scientific_config_semantic_sha256"
                ),
                "formal_training_execution_binding_sha256": row.get(
                    "formal_training_execution_binding_sha256"
                ),
                "execution_commit": row.get("execution_commit"),
                "resolved_execution_context_sha256": row.get(
                    "resolved_execution_context_sha256"
                ),
                "formal_agent_order_contract_semantic_sha256": row.get(
                    "formal_agent_order_contract_semantic_sha256"
                ),
                "active_formal_bundle_sha256": row.get(
                    "active_formal_bundle_sha256"
                ),
                "formal_nullable_metric_aggregation_contract_semantic_sha256": row.get(
                    "formal_nullable_metric_aggregation_contract_semantic_sha256"
                ),
                "runtime_contract_sha256": row.get("runtime_contract_sha256"),
                "resolved_agent_config": row.get("resolved_agent_config"),
                "checkpoint_schedule": row.get("checkpoint_schedule"),
                "selection_sha256": freeze["selection_sha256"],
                "checkpoint_identity": row["checkpoint_identity"],
                "artifact_location": row["artifact_location"],
            }
        target = input_root / "checkpoint_manifests" / capacity_label
        target.mkdir(parents=True, exist_ok=True)
        seed_path = target / "seed_checkpoint_manifest.json"
        provenance_path = target / "checkpoint_provenance_manifest.json"
        seed_payload: dict[str, Any] = {
            "_portable_checkpoint_manifest": {
                "checkpoint_manifest_version": "1.1.0",
                "checkpoint_location_contract_version": "1.0.0",
                "manifest_id": f"checkpoint-manifest-{capacity_label}",
                "protocol_semantic_sha256": freeze["protocol_semantic_sha256"],
                "formal_training_identity": freeze.get("formal_training_identity"),
                "formal_agent_order_contract_semantic_sha256": freeze.get(
                    "formal_agent_order_contract_semantic_sha256"
                ),
                "active_formal_bundle_sha256": (
                    freeze.get("formal_training_identity", {}) or {}
                ).get("active_formal_bundle_sha256"),
                "entries": [
                    {
                        "agent": row["agent_name"],
                        "seed": int(row["seed"]),
                        "checkpoint_identity": row["checkpoint_identity"],
                        "artifact_location": row["artifact_location"],
                    }
                    for row in rows
                ],
                "invalid_run_roots": [str(root) for root in INVALID_FORMAL_RUN_ROOTS],
            },
            **seed_manifest,
        }
        write_create_only(seed_path, seed_payload)
        write_create_only(provenance_path, provenance_manifest)
        outputs.append(
            {
                "capacity_label": capacity_label,
                "seed_checkpoint_manifest_path": str(seed_path.resolve()),
                "checkpoint_provenance_manifest_path": str(provenance_path.resolve()),
                "checkpoint_count": len(rows),
            }
        )
    return outputs


def formal_gate(input_root: Path, protocol: dict) -> dict:
    rehearsal_marker = input_root / "non_formal_rehearsal.json"
    non_formal_rehearsal = rehearsal_marker.is_file()
    required = [
        "checkpoint_freeze.json",
        "formal_cache_policy/**/aggregate_summary.json",
        "formal_controller/**/aggregate_summary.json",
        "formal_ablation/**/support_provenance.json",
        "formal_support/**/support_provenance.json",
        "formal_scalability/**/support_provenance.json",
        "statistics/paired_statistics.json",
        "artifact_integrity_manifest.json",
    ]
    missing = [pattern for pattern in required if not any(input_root.glob(pattern))]
    endpoint_availability: dict[str, str] = {}
    comparison_availability: list[dict[str, Any]] = []
    statistics_path = input_root / "statistics" / "paired_statistics.json"
    if statistics_path.is_file():
        statistics_payload = read_json(statistics_path)
        for row in statistics_payload.get("rows", []):
            if not isinstance(row, dict):
                raise ValueError("statistics comparison row must be an object")
            status = (
                "AVAILABLE"
                if int(row.get("available_paired_count", row.get("paired_count", 0))) > 0
                else "UNAVAILABLE"
            )
            metric = str(row.get("metric"))
            existing = endpoint_availability.get(metric)
            endpoint_availability[metric] = (
                "AVAILABLE" if existing == "AVAILABLE" or status == "AVAILABLE" else status
            )
            comparison_availability.append(
                {
                    "candidate_agent": row.get("candidate_agent"),
                    "baseline_agent": row.get("baseline_agent"),
                    "metric": metric,
                    "status": status,
                    "available_paired_count": int(
                        row.get("available_paired_count", row.get("paired_count", 0))
                    ),
                    "claim_interpretation": (
                        "eligible_for_statistical_interpretation"
                        if status == "AVAILABLE"
                        else "UNAVAILABLE_not_tie_pass_or_fail"
                    ),
                }
            )
    return {
        "formal_execution_gate_version": "1.1.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "passed": not missing,
        "missing_outputs": missing,
        "performance_threshold_used": False,
        "holdout_opened": False,
        "completeness_only": True,
        "endpoint_availability": endpoint_availability,
        "claim_map_availability": comparison_availability,
        "zero_pair_rule": "UNAVAILABLE_not_tie_pass_or_fail",
        "execution_mode": (
            "non_formal_rehearsal" if non_formal_rehearsal else "formal"
        ),
        "formal_execution_started": not non_formal_rehearsal,
        "paper_claims_permitted": False if non_formal_rehearsal else None,
    }


def artifact_integrity(input_root: Path, protocol: dict, output_path: Path) -> dict:
    non_formal_rehearsal = (input_root / "non_formal_rehearsal.json").is_file()
    files = []
    for path in sorted(input_root.rglob("*")):
        if (
            not path.is_file()
            or path.resolve() == output_path.resolve()
            or path.relative_to(input_root).as_posix() == "phase_state.jsonl"
        ):
            continue
        digest = sha256_file(path)
        files.append(
            {
                "path": path.relative_to(input_root).as_posix(),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "formal_artifact_integrity_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "file_count": len(files),
        "files": files,
        "integrity_sha256": canonical_sha256(files),
        "excluded_live_files": [
            "phase_state.jsonl (the phase runner appends formal_gate and complete_without_holdout after this scan)"
        ],
        "performance_threshold_used": False,
        "holdout_opened": False,
        "execution_mode": (
            "non_formal_rehearsal" if non_formal_rehearsal else "formal"
        ),
        "formal_execution_started": not non_formal_rehearsal,
    }


def main() -> None:
    args = parse_args()
    try:
        reject_permanently_invalid_formal_references(
            [args.input_root, args.output_path]
        )
    except PermanentlyInvalidFormalReferenceError as exc:
        raise ValueError(str(exc)) from exc
    protocol = read_json(Path(args.protocol_path))
    validate_protocol_v1_1(protocol)
    input_root = Path(args.input_root)
    output_path = Path(args.output_path)
    if args.action == "dev_select":
        payload = dev_select(input_root, protocol)
    elif args.action == "checkpoint_freeze":
        payload = checkpoint_freeze(input_root, protocol)
        payload["checkpoint_companions"] = write_checkpoint_companions(
            input_root, payload
        )
    elif args.action == "integrity":
        payload = artifact_integrity(input_root, protocol, output_path)
    elif args.action == "integrity_and_formal_gate":
        integrity_path = input_root / "artifact_integrity_manifest.json"
        write_create_only(
            integrity_path,
            artifact_integrity(input_root, protocol, integrity_path),
        )
        payload = formal_gate(input_root, protocol)
    else:
        payload = formal_gate(input_root, protocol)
    write_create_only(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
