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


def dev_select(input_root: Path, protocol: dict) -> dict:
    candidates = read_json(input_root / "checkpoint_candidates.json")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("checkpoint_candidates.json must contain a non-empty list")
    selected = []
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("checkpoint candidate must be an object")
        key = (str(row["agent_name"]), int(row["seed"]), str(row["capacity_label"]))
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items()):
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
    return {
        "dev_checkpoint_selection_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_split": "dev",
        "formal_or_holdout_used": False,
        "selection_rule": protocol["training_budget"]["checkpoint_selection"]["metric_rule"],
        "selected": selected,
        "selection_sha256": canonical_sha256(selected),
    }


def checkpoint_freeze(input_root: Path, protocol: dict) -> dict:
    selection = read_json(input_root / "dev_selection.json")
    if selection.get("protocol_semantic_sha256") != protocol["hashes"]["semantic_sha256"]:
        raise ValueError("dev selection protocol hash mismatch")
    frozen = []
    for row in selection.get("selected", []):
        path = Path(row["checkpoint_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != row.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint hash mismatch: {path}")
        frozen.append({**row, "checkpoint_path": str(path.resolve())})
    return {
        "checkpoint_freeze_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "frozen_checkpoint_count": len(frozen),
        "frozen_checkpoints": frozen,
        "formal_or_holdout_used": False,
        "freeze_sha256": canonical_sha256(frozen),
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
                "runtime_contract_sha256": row.get("runtime_contract_sha256"),
                "resolved_agent_config": row.get("resolved_agent_config"),
                "checkpoint_schedule": row.get("checkpoint_schedule"),
                "selection_sha256": freeze["selection_sha256"],
            }
        target = input_root / "checkpoint_manifests" / capacity_label
        target.mkdir(parents=True, exist_ok=True)
        seed_path = target / "seed_checkpoint_manifest.json"
        provenance_path = target / "checkpoint_provenance_manifest.json"
        write_create_only(seed_path, seed_manifest)
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
    }


def artifact_integrity(input_root: Path, protocol: dict, output_path: Path) -> dict:
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
