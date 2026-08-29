"""Outcome-blind dev selection, checkpoint freeze, and completeness gate."""

from __future__ import annotations

import argparse
import json
import math
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
INVALID_FORMAL_RUN_ROOTS = (
    *INVALID_G14C_V1_V2_RUN_ROOTS,
    INVALID_G14C_V3_RUN_ROOT,
    *INVALID_G14C_V4_RUN_ROOTS,
    INVALID_G14C_V5_RUN_ROOT,
    INVALID_G14C_V6_RUN_ROOT,
    INVALID_G14C_V7_RUN_ROOT,
    INVALID_G14C_V8_RUN_ROOT,
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


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite dev endpoint: {field}")
    return result


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
    if protocol["typed_model_cache_formal_protocol_version"] in {"1.7.0", "1.8.0", "1.9.0"}:
        try:
            order_audit = resolve_formal_agent_order(protocol=protocol)
            reject_permanently_invalid_run_references([input_root])
        except FormalAgentOrderError as exc:
            raise ValueError(str(exc)) from exc
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("checkpoint candidate must be an object")
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
        ordered_group_keys = [
            (agent, seed, capacity)
            for capacity in capacities
            for agent in order_audit["learned_agent_order"]
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
        ranked = sorted(
            rows,
            key=lambda row: (
                -finite(row["full_service_ready_byte_hit_rate"], "full_service_ready_byte_hit_rate"),
                -finite(row["workflow_continuity_rate"], "workflow_continuity_rate"),
                finite(row["transfer_mb_per_request"], "transfer_mb_per_request"),
                finite(row["end_to_end_workflow_delay"], "end_to_end_workflow_delay"),
                int(row["update_index"]),
                str(row["checkpoint_sha256"]),
            ),
        )
        winner = dict(ranked[0])
        selected.append(winner)
    training_identity_fields = (
        "agent_scientific_config_semantic_sha256",
        "formal_training_execution_binding_sha256",
        "formal_protocol_semantic_sha256",
        "execution_commit",
        "resolved_execution_context_sha256",
    )
    if protocol["typed_model_cache_formal_protocol_version"] in {"1.8.0", "1.9.0"}:
        training_identity_fields = (*training_identity_fields, "active_formal_bundle_sha256")
    training_identity = None
    if (
        not non_formal_rehearsal
        and protocol["typed_model_cache_formal_protocol_version"] in {"1.6.0", "1.7.0", "1.8.0", "1.9.0"}
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
    return {
        "dev_checkpoint_selection_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_split": "dev",
        "formal_or_holdout_used": False,
        "selection_rule": protocol["training_budget"]["checkpoint_selection"]["metric_rule"],
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
            order_audit["learned_agent_order"] if order_audit else None
        ),
        "non_formal_rehearsal": non_formal_rehearsal,
    }


def checkpoint_freeze(input_root: Path, protocol: dict) -> dict:
    selection = read_json(input_root / "dev_selection.json")
    if selection.get("protocol_semantic_sha256") != protocol["hashes"]["semantic_sha256"]:
        raise ValueError("dev selection protocol hash mismatch")
    formal_training_identity = selection.get("formal_training_identity")
    order_audit = None
    if protocol["typed_model_cache_formal_protocol_version"] in {"1.7.0", "1.8.0", "1.9.0"}:
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
        if observed_selected_agents != order_audit["learned_agent_order"]:
            raise ValueError("dev selection learned-agent order drift")
    if (
        not bool(selection.get("non_formal_rehearsal"))
        and protocol["typed_model_cache_formal_protocol_version"] in {"1.6.0", "1.7.0", "1.8.0", "1.9.0"}
    ):
        if not isinstance(formal_training_identity, dict):
            raise ValueError("dev selection lacks formal training identity")
        if formal_training_identity.get("formal_protocol_semantic_sha256") != protocol["hashes"]["semantic_sha256"]:
            raise ValueError("dev selection formal training Protocol identity mismatch")
    frozen = []
    for row in selection.get("selected", []):
        path = Path(row["checkpoint_path"]).resolve()
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
    return {
        "formal_execution_gate_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "passed": not missing,
        "missing_outputs": missing,
        "performance_threshold_used": False,
        "holdout_opened": False,
        "completeness_only": True,
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
