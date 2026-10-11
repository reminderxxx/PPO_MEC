"""Run one bounded, non-training event-label coverage probe on Lankershim."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cscwd_phase_v2_causal_public_pilot import (  # noqa: E402
    _load_runtime,
    _phase_env,
)
from src.agents.causal_public_action_estimator import (  # noqa: E402
    public_prepare_advantage_label,
)


DEFAULT_PROTOCOL = (
    ROOT / "configs/experiment/cscwd_lankershim_event_label_coverage_probe_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/analysis/cscwd_lankershim_event_label_coverage_probe_20261011_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _event_label_diagnostics(
    semantic_state: dict[str, Any], action_mask: list[bool], *,
    positive_prepare_calibration_abstention_enabled: bool = False,
) -> dict[str, Any]:
    """Flatten public-only label inputs without changing the label or policy."""

    label = public_prepare_advantage_label(
        semantic_state,
        action_mask,
        positive_prepare_calibration_abstention_enabled=(
            positive_prepare_calibration_abstention_enabled
        ),
    )
    estimate = label["estimate"]
    action4 = estimate["actions"]["4"]
    service_actions = [
        estimate["actions"][str(action)]
        for action in (0, 2, 3)
        if estimate["actions"][str(action)]["legal"]
    ]
    canonical_state = json.dumps(
        semantic_state, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance = dict(
        dict(semantic_state.get("predictions", {}) or {}).get(
            "causal_provenance", {}
        )
        or {}
    )
    calibrated = provenance.get("confidence_calibrated")
    return {
        "event_label_schema_version": label["schema_version"],
        "event_label": label["decision"],
        "event_label_reason": label["reason"],
        "event_supervision_weight": float(label["supervision_weight"]),
        "semantic_state_sha256": hashlib.sha256(canonical_state).hexdigest(),
        "action_mask": json.dumps([bool(value) for value in action_mask]),
        "prediction_confidence_calibrated_present": (
            "confidence_calibrated" in provenance
        ),
        "prediction_confidence_calibrated_literal_true": calibrated is True,
        "prediction_confidence_calibrated_value_type": type(calibrated).__name__,
        "action4_legal": bool(action4["legal"]),
        "action4_current_service": action4["current_service"],
        "action4_target_prepare": action4["target_prepare"],
        "action4_raw_full_step_contact_fit": action4["raw_full_step_contact_fit"],
        "action4_deadline_fit": action4["deadline_fit"],
        "action4_future_readiness_gain": action4["future_readiness_gain"],
        "service_action_current_and_deadline_fit": any(
            row["current_service"] == "yes" and row["deadline_fit"] == "yes"
            for row in service_actions
        ),
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "cscwd_lankershim_event_label_coverage_probe_v1",
        "execution_authorized": True,
        "authorization_scope": "one_create_only_nontraining_probe_no_retry",
        "source_segment_id": "lankershim",
        "condition_id": "default",
        "workload_design_id": "dev_01",
        "seed": 7,
        "environment_step_budget": 96,
        "episode_step_cap": 24,
        "desired_action_cycle": [0, 1, 2, 3, 4],
        "masked_action_fallback": "lowest_legal_action",
        "selection_is_result_blind": True,
        "training_enabled": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "checkpoint_writes": 0,
        "formal_or_holdout_reads": 0,
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"probe protocol drift: {key}")
    source_path = (ROOT / protocol["source_protocol"]).resolve()
    if _sha256(source_path) != protocol["source_protocol_sha256"]:
        raise RuntimeError("source protocol identity drift")
    source_protocol = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source_protocol["data_assignment"]["training_source_segment"]
        != protocol["source_segment_id"]
        or source_protocol["data_assignment"]["training_condition"]
        != protocol["condition_id"]
        or source_protocol["data_assignment"]["workload_design_id"]
        != protocol["workload_design_id"]
    ):
        raise RuntimeError("probe differs from frozen training domain")
    return protocol


def _select_requested_action(
    desired_action: int, action_mask: list[bool],
) -> tuple[int, bool]:
    if 0 <= desired_action < len(action_mask) and action_mask[desired_action]:
        return desired_action, False
    legal = [index for index, allowed in enumerate(action_mask) if allowed]
    if not legal:
        raise RuntimeError("probe encountered an empty action mask")
    return legal[0], True


def _preflight(
    protocol_path: Path, protocol: dict[str, Any], output_root: Path,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    if not _git_clean():
        raise RuntimeError("probe checkout must be clean")
    source_protocol_path = (ROOT / protocol["source_protocol"]).resolve()
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    runtime = _load_runtime(source_protocol)
    config, instance, source_bundle, runtime_info = runtime
    env = _phase_env(
        config,
        instance,
        source_bundle["traces"][protocol["source_segment_id"]],
        runtime_info["phase"],
        protocol["condition_id"],
    )
    _, info = env.reset()
    if env.__class__.__name__ != "RawNGSIMPhaseV2Env":
        raise RuntimeError("actual probe environment class drift")
    diagnostics = _event_label_diagnostics(
        info["semantic_state"], list(info["action_mask"])
    )
    receipt = {
        "status": "PASS",
        "mode": "preflight_zero_environment_steps",
        "git_commit": _git_commit(),
        "git_clean": True,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "source_protocol_sha256": _sha256(source_protocol_path),
        "environment_class": env.__class__.__name__,
        "source_segment_id": protocol["source_segment_id"],
        "condition_id": protocol["condition_id"],
        "initial_semantic_state_sha256": diagnostics["semantic_state_sha256"],
        "environment_steps": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "formal_or_holdout_reads": 0,
    }
    return receipt, runtime


def run(protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    preflight, runtime = _preflight(protocol_path, protocol, output_root)
    config, instance, source_bundle, runtime_info = runtime
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    env_factory = lambda: _phase_env(
        config,
        instance,
        source_bundle["traces"][protocol["source_segment_id"]],
        runtime_info["phase"],
        protocol["condition_id"],
    )
    env = env_factory()
    observation, info = env.reset()
    del observation
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    episode_index = 1
    for global_step in range(1, protocol["environment_step_budget"] + 1):
        desired = protocol["desired_action_cycle"][
            (global_step - 1) % len(protocol["desired_action_cycle"])
        ]
        mask = [bool(value) for value in info["action_mask"]]
        requested, fallback = _select_requested_action(desired, mask)
        semantic_state = deepcopy(info["semantic_state"])
        base = _event_label_diagnostics(semantic_state, mask)
        gated = _event_label_diagnostics(
            semantic_state,
            mask,
            positive_prepare_calibration_abstention_enabled=True,
        )
        states.append({
            "global_step": global_step,
            "episode_index": episode_index,
            "episode_step_before_action": int(env.step_index),
            "semantic_state_sha256": base["semantic_state_sha256"],
            "action_mask": mask,
            "semantic_state": semantic_state,
        })
        _, reward, terminated, truncated, next_info = env.step(requested)
        transition = dict(next_info.get("transition", {}) or {})
        executed = int(transition.get("action", requested))
        if executed != requested:
            raise RuntimeError("probe action was rewritten")
        rows.append({
            "global_step": global_step,
            "episode_index": episode_index,
            "episode_step_before_action": int(env.step_index) - 1,
            "semantic_state_sha256": base["semantic_state_sha256"],
            "desired_action": desired,
            "requested_action": requested,
            "executed_action": executed,
            "masked_fallback_used": fallback,
            "reward_observed_not_used_for_selection": float(reward),
            "terminated": bool(terminated),
            "environment_truncated": bool(truncated),
            "base_event_label": base["event_label"],
            "base_event_label_reason": base["event_label_reason"],
            "base_event_supervision_weight": base["event_supervision_weight"],
            "gated_event_label": gated["event_label"],
            "gated_event_label_reason": gated["event_label_reason"],
            "gated_event_supervision_weight": gated["event_supervision_weight"],
            "prediction_confidence_calibrated_present": base[
                "prediction_confidence_calibrated_present"
            ],
            "prediction_confidence_calibrated_literal_true": base[
                "prediction_confidence_calibrated_literal_true"
            ],
            "action4_legal": base["action4_legal"],
            "action4_current_service": base["action4_current_service"],
            "action4_target_prepare": base["action4_target_prepare"],
            "action4_raw_full_step_contact_fit": base[
                "action4_raw_full_step_contact_fit"
            ],
            "action4_deadline_fit": base["action4_deadline_fit"],
            "action4_future_readiness_gain": base[
                "action4_future_readiness_gain"
            ],
            "service_action_current_and_deadline_fit": base[
                "service_action_current_and_deadline_fit"
            ],
            "service_completed": bool(transition.get("service_completed")),
            "failure_reason": transition.get("failure_reason") or "",
        })
        outer_cap = (
            not terminated
            and not truncated
            and int(env.step_index) >= protocol["episode_step_cap"]
        )
        if terminated or truncated or outer_cap:
            env = env_factory()
            _, info = env.reset()
            episode_index += 1
        else:
            info = next_info

    base_labels = Counter(row["base_event_label"] for row in rows)
    gated_labels = Counter(row["gated_event_label"] for row in rows)
    base_reasons = Counter(row["base_event_label_reason"] for row in rows)
    summary = {
        "schema_version": "cscwd_lankershim_event_label_coverage_probe_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "interpretation": (
            "bounded mechanism coverage only; an unobserved label is not proof "
            "that the training domain can never produce it"
        ),
        "git_commit": preflight["git_commit"],
        "protocol_sha256": preflight["protocol_sha256"],
        "source_segment_id": protocol["source_segment_id"],
        "condition_id": protocol["condition_id"],
        "environment_steps": len(rows),
        "episodes_started": episode_index,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "checkpoint_writes": 0,
        "formal_or_holdout_reads": 0,
        "base_event_label_counts": dict(sorted(base_labels.items())),
        "gated_event_label_counts": dict(sorted(gated_labels.items())),
        "base_event_reason_counts": dict(sorted(base_reasons.items())),
        "desired_action_counts": dict(sorted(Counter(
            str(row["desired_action"]) for row in rows
        ).items())),
        "executed_action_counts": dict(sorted(Counter(
            str(row["executed_action"]) for row in rows
        ).items())),
        "masked_fallback_count": sum(row["masked_fallback_used"] for row in rows),
        "unique_semantic_state_count": len({
            row["semantic_state_sha256"] for row in rows
        }),
        "positive_prepare_observed": base_labels["prepare"] > 0,
        "serve_observed": base_labels["serve"] > 0,
    }
    output_root.mkdir(parents=True)
    _write_json(output_root / "preflight_receipt.json", preflight)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "probe_rows.csv", rows)
    with (output_root / "semantic_states.jsonl").open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    files = []
    for path in sorted(output_root.iterdir()):
        if path.name != "artifact_integrity.json":
            files.append({
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    _write_json(output_root / "artifact_integrity.json", {
        "schema_version": "cscwd_lankershim_event_label_coverage_probe_integrity_v1",
        "files": files,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_root = args.output.resolve()
    protocol = _load_protocol(protocol_path)
    if args.preflight:
        receipt, _ = _preflight(protocol_path, protocol, output_root)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return
    print(json.dumps(run(protocol_path, output_root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
