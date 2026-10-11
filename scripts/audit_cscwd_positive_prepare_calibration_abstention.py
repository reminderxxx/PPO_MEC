"""Statically recompute the positive-prepare calibration-abstention labels."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.causal_public_action_estimator import (  # noqa: E402
    CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION,
    EVENT_LABEL_SCHEMA_VERSION,
    public_prepare_advantage_label,
)


SCHEMA_VERSION = "cscwd_positive_prepare_calibration_abstention_static_audit_v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_info(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("info"), dict):
        return payload["info"]
    if isinstance(payload.get("policy_input_info"), dict):
        return payload["policy_input_info"]
    if "semantic_state" in payload and "action_mask" in payload:
        return payload
    raise ValueError("input does not contain public decision info")


def _calibration_class(value: Any, *, present: bool) -> str:
    if not present:
        return "missing"
    if value is True:
        return "literal_true"
    if value is False:
        return "literal_false"
    return f"nonliteral_{type(value).__name__}"


def _recompute(path: Path) -> dict[str, Any]:
    info = _public_info(_load(path))
    semantic_state = info["semantic_state"]
    action_mask = info["action_mask"]
    provenance = (semantic_state.get("predictions") or {}).get("causal_provenance") or {}
    present = "confidence_calibrated" in provenance
    old = public_prepare_advantage_label(semantic_state, action_mask)
    new = public_prepare_advantage_label(
        semantic_state,
        action_mask,
        positive_prepare_calibration_abstention_enabled=True,
    )
    if old["schema_version"] != EVENT_LABEL_SCHEMA_VERSION:
        raise RuntimeError("legacy label schema drift")
    if new["schema_version"] != CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION:
        raise RuntimeError("gated label schema drift")
    allowed = {
        ("serve", "serve"),
        ("abstain", "abstain"),
        ("prepare", "prepare"),
        ("prepare", "abstain"),
    }
    transition = (str(old["decision"]), str(new["decision"]))
    if transition not in allowed:
        raise RuntimeError(f"unexpected label transition: {transition}")
    if old["decision"] == "serve" and new != {
        **old,
        "schema_version": CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION,
    }:
        raise RuntimeError("serve-label semantics changed")
    if old["estimate"] != new["estimate"]:
        raise RuntimeError("calibration gate changed the public estimator")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "calibration_provenance": _calibration_class(
            provenance.get("confidence_calibrated"), present=present
        ),
        "legacy_decision": old["decision"],
        "legacy_target": old["event_target"],
        "legacy_supervision_weight": old["supervision_weight"],
        "gated_decision": new["decision"],
        "gated_target": new["event_target"],
        "gated_supervision_weight": new["supervision_weight"],
        "gated_reason": new["reason"],
    }


def _raw_training_coverage(manifest_path: Path, producer_path: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    producer_text = producer_path.read_text(encoding="utf-8")
    literal = '"confidence_calibrated": False'
    if literal not in producer_text:
        raise RuntimeError("raw producer no longer emits literal false calibration provenance")
    counts = manifest.get("counts") or {}
    training_steps = int(counts.get("training_steps", 0))
    if training_steps <= 0:
        raise RuntimeError("pilot manifest has no training steps")
    return {
        "classification": "contract_proven_all_raw_training_states_uncalibrated",
        "pilot_run_id": Path(manifest_path).parent.name,
        "pilot_manifest_path": str(manifest_path),
        "pilot_manifest_sha256": _sha256(manifest_path),
        "pilot_git_commit": manifest.get("git_commit"),
        "training_steps": training_steps,
        "learned_cells": int(counts.get("learned_cells", 0)),
        "raw_producer_path": str(producer_path),
        "raw_producer_sha256": _sha256(producer_path),
        "producer_contract_literal": literal,
        "per_transition_semantic_state_persisted": False,
        "evidence_boundary": (
            "source-contract proof for the named pilot, not a row-by-row replay of "
            "all historical semantic states"
        ),
        "candidate_effect_on_raw_profile": (
            "all positive prepare supervision becomes abstention; serve supervision "
            "and PPO/slow/fast losses are unchanged"
        ),
        "fine_grained_uncertainty_control_demonstrated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-info", action="append", default=[])
    parser.add_argument("--snapshot-dir", action="append", default=[])
    parser.add_argument("--pilot-run-manifest", required=True)
    parser.add_argument(
        "--raw-producer",
        default="src/envs/core/raw_ngsim_event_time_env.py",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_paths = [Path(value).resolve() for value in args.public_info]
    for value in args.snapshot_dir:
        input_paths.extend(sorted(Path(value).resolve().glob("*.json")))
    if not input_paths:
        raise RuntimeError("at least one public-info input is required")
    if len(set(input_paths)) != len(input_paths):
        raise RuntimeError("duplicate public-info inputs")

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    rows = [_recompute(path) for path in input_paths]
    transitions = Counter(
        f"{row['legacy_decision']}->{row['gated_decision']}" for row in rows
    )
    legacy = Counter(str(row["legacy_decision"]) for row in rows)
    gated = Counter(str(row["gated_decision"]) for row in rows)
    calibration = Counter(str(row["calibration_provenance"]) for row in rows)
    changed = [row for row in rows if row["legacy_decision"] != row["gated_decision"]]
    if any(
        row["legacy_decision"] != "prepare"
        or row["gated_decision"] != "abstain"
        or float(row["gated_supervision_weight"]) != 0.0
        for row in changed
    ):
        raise RuntimeError("gate changed more than positive prepare supervision")

    manifest_path = Path(args.pilot_run_manifest).resolve()
    producer_path = Path(args.raw_producer)
    if not producer_path.is_absolute():
        producer_path = (ROOT / producer_path).resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "static_label_recompute_only",
        "environment_steps": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "formal_or_holdout_reads": 0,
        "input_count": len(rows),
        "legacy_decisions": dict(sorted(legacy.items())),
        "gated_decisions": dict(sorted(gated.items())),
        "transition_matrix": dict(sorted(transitions.items())),
        "calibration_provenance": dict(sorted(calibration.items())),
        "changed_count": len(changed),
        "all_changes_prepare_to_abstain": True,
        "serve_labels_preserved": True,
        "public_estimates_preserved": True,
        "raw_training_coverage": _raw_training_coverage(
            manifest_path, producer_path
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "input_count": len(rows),
        "transition_matrix": payload["transition_matrix"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
