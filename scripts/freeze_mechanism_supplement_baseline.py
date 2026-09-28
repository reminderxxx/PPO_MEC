#!/usr/bin/env python3
"""Create a read-only manifest for the fixed v2 mechanism checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
FREEZE_VERSION = "mechanism_supplement_checkpoint_freeze_v1"
EXPECTED_AGENTS = ("sa_ghmappo", "mappo")
FIXED_UPDATE_INDEX = 16


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT_DIR, text=True
    ).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_state_digest(state: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
        tensor_count += 1
        element_count += int(value.numel())
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "tensor_state_sha256": digest.hexdigest(),
    }


def _summary_path(training_root: Path, agent_name: str) -> Path:
    candidates = sorted((training_root / "runs" / agent_name).glob("*/summary.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one summary for {agent_name}, found {len(candidates)}"
        )
    return candidates[0]


def freeze(training_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse freeze output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    completion_path = training_root / "completion_receipt.json"
    command_manifest_path = training_root / "command_manifest.json"
    completion = _read_json(completion_path)
    command_manifest = _read_json(command_manifest_path)
    if completion.get("status") != "SUCCEEDED" or completion.get("return_code") != 0:
        raise ValueError("training root is not a successful completed run")
    manifest_sha256 = _file_sha256(command_manifest_path)
    if manifest_sha256 != str(completion.get("command_manifest_sha256")):
        raise ValueError("completion receipt command manifest SHA-256 mismatch")
    if tuple(command_manifest.get("agents", [])) != EXPECTED_AGENTS:
        raise ValueError("unexpected agent order in command manifest")

    checkpoints: list[dict[str, Any]] = []
    exposure_by_agent: dict[str, list[str]] = {}
    for agent_name in EXPECTED_AGENTS:
        summary_path = _summary_path(training_root, agent_name)
        summary = _read_json(summary_path)
        if int(summary.get("update_count", -1)) != FIXED_UPDATE_INDEX:
            raise ValueError(f"{agent_name} did not reach update {FIXED_UPDATE_INDEX}")
        expected_name = f"update_{FIXED_UPDATE_INDEX:04d}.pt"
        matches = [
            Path(str(path))
            for path in summary.get("checkpoint_paths", [])
            if Path(str(path)).name == expected_name
        ]
        if len(matches) != 1:
            raise ValueError(f"{agent_name} lacks exactly one {expected_name}")
        checkpoint_path = matches[0].resolve()
        before_sha256 = _file_sha256(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint payload is not a mapping: {checkpoint_path}")
        if str(payload.get("agent_name")) != agent_name:
            raise ValueError(f"checkpoint agent mismatch: {checkpoint_path}")
        if int(payload.get("update_count", -1)) != FIXED_UPDATE_INDEX:
            raise ValueError(f"checkpoint update mismatch: {checkpoint_path}")
        network_state = payload.get("network_state_dict")
        if not isinstance(network_state, dict) or not network_state:
            raise ValueError(f"checkpoint lacks network_state_dict: {checkpoint_path}")
        tensor_audit = _tensor_state_digest(network_state)
        del payload
        after_sha256 = _file_sha256(checkpoint_path)
        if before_sha256 != after_sha256:
            raise RuntimeError("checkpoint changed during read-only load validation")
        latest_path = Path(str(summary["latest_checkpoint_path"])).resolve()
        exposure_fingerprints = [str(item) for item in summary["request_exposure_fingerprints"]]
        exposure_by_agent[agent_name] = exposure_fingerprints
        run_dir = summary_path.parent
        checkpoints.append(
            {
                "agent_name": agent_name,
                "selection_role": "fixed_budget_terminal_update",
                "fixed_update_index": FIXED_UPDATE_INDEX,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": before_sha256,
                "latest_path": str(latest_path),
                "latest_sha256": _file_sha256(latest_path),
                "latest_matches_fixed_checkpoint": _file_sha256(latest_path)
                == before_sha256,
                "read_only_load_validation": {
                    "status": "passed",
                    "agent_name": agent_name,
                    "update_count": FIXED_UPDATE_INDEX,
                    "file_sha256_before": before_sha256,
                    "file_sha256_after": after_sha256,
                    "file_unchanged": True,
                    **tensor_audit,
                },
                "summary": {
                    "path": str(summary_path.resolve()),
                    "size_bytes": summary_path.stat().st_size,
                    "sha256": _file_sha256(summary_path),
                },
                "train_csv": {
                    "path": str((run_dir / "train.csv").resolve()),
                    "size_bytes": (run_dir / "train.csv").stat().st_size,
                    "sha256": _file_sha256(run_dir / "train.csv"),
                },
                "train_summary": {
                    "path": str((run_dir / "train_summary.json").resolve()),
                    "size_bytes": (run_dir / "train_summary.json").stat().st_size,
                    "sha256": _file_sha256(run_dir / "train_summary.json"),
                },
            }
        )
    if exposure_by_agent[EXPECTED_AGENTS[0]] != exposure_by_agent[EXPECTED_AGENTS[1]]:
        raise ValueError("agent request exposure sequences do not match")

    freeze_manifest = {
        "freeze_version": FREEZE_VERSION,
        "created_at": _utc_now(),
        "status": "frozen",
        "create_only": True,
        "source_training_root": str(training_root),
        "source_training_root_mutated": False,
        "selection_rule": {
            "checkpoint": "common fixed-budget terminal update_0016.pt",
            "performance_based_selection": False,
            "rule_timing": "adopted_after_training_statistics_were_viewed",
            "preregistered": False,
            "claim_boundary": (
                "The rule avoids choosing among candidate checkpoints by observed "
                "performance, but it was not preregistered and must not be described as such."
            ),
        },
        "training_scope": command_manifest.get("evidence_scope"),
        "formal": False,
        "holdout": False,
        "independent_evaluation": False,
        "training_git_commit": command_manifest.get("git_commit"),
        "training_git_tree": command_manifest.get("git_tree"),
        "freezer_git_commit": _git_value("rev-parse", "HEAD"),
        "freezer_git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "command_manifest": {
            "path": str(command_manifest_path.resolve()),
            "size_bytes": command_manifest_path.stat().st_size,
            "sha256": manifest_sha256,
        },
        "completion_receipt": {
            "path": str(completion_path.resolve()),
            "size_bytes": completion_path.stat().st_size,
            "sha256": _file_sha256(completion_path),
        },
        "config": command_manifest.get("config_path"),
        "config_sha256": command_manifest.get("config_sha256"),
        "runtime_config": command_manifest.get("runtime_config"),
        "window_plan": command_manifest.get("window_plan"),
        "sources": command_manifest.get("sources"),
        "budget": command_manifest.get("budget"),
        "python_identity": command_manifest.get("python_identity"),
        "request_exposure_sequence_sha256": _canonical_sha256(
            exposure_by_agent[EXPECTED_AGENTS[0]]
        ),
        "request_exposure_count": len(exposure_by_agent[EXPECTED_AGENTS[0]]),
        "request_exposure_sequences_match": True,
        "checkpoints": checkpoints,
        "claim_boundary": (
            "Frozen observed-data supplement baseline only; not formal, holdout, "
            "independent evaluation, convergence evidence, or algorithm superiority."
        ),
    }
    manifest_path = output_dir / "freeze_manifest.json"
    _write_json(manifest_path, freeze_manifest)
    receipt = {
        "freeze_version": FREEZE_VERSION,
        "status": "completed",
        "freeze_manifest_path": str(manifest_path.resolve()),
        "freeze_manifest_sha256": _file_sha256(manifest_path),
        "checkpoint_count": len(checkpoints),
        "source_training_root_mutated": False,
    }
    _write_json(output_dir / "completion_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = freeze(args.training_root.resolve(), args.output_dir.resolve())
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
