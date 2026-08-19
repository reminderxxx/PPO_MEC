"""Build a G10 read-only cache information sufficiency audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.information_sufficiency_audit import (  # noqa: E402
    INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION,
    audit_information_sufficiency,
    build_synthetic_validation_report,
)
from src.evaluators.cache_baseline_fairness import load_and_validate_manifest  # noqa: E402
from src.oracles.cache_request_replay import load_and_validate_request_replay  # noqa: E402


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cache decision information sufficiency and MARL necessity")
    parser.add_argument("--fairness_manifest_path", required=True)
    parser.add_argument("--request_replay_path", required=True)
    parser.add_argument("--oracle_action_trace_path", required=True)
    parser.add_argument("--opportunity_rows_path", required=True)
    parser.add_argument("--observation_trace_path")
    parser.add_argument("--agent_identity", choices=["ppo", "mappo", "sa_ghmappo"], required=True)
    parser.add_argument("--audit_config_path")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G10 audit directory: {output}")
    manifest, manifest_report = load_and_validate_manifest(args.fairness_manifest_path, root=ROOT)
    replay, replay_report = load_and_validate_request_replay(args.request_replay_path, source_manifest=manifest)
    oracle_trace = _load(args.oracle_action_trace_path)
    opportunity_rows = _load(args.opportunity_rows_path)
    observation_trace = _load(args.observation_trace_path) if args.observation_trace_path else None
    config = _load(args.audit_config_path) if args.audit_config_path else None
    bundle = audit_information_sufficiency(
        manifest=manifest,
        replay=replay,
        oracle_action_trace=oracle_trace,
        opportunity_rows_payload=opportunity_rows,
        observation_trace=observation_trace,
        agent_identity=args.agent_identity,
        config=config,
    )
    output.mkdir(parents=True)
    identity = bundle["identity"]
    identity["artifact_run_id"] = output.name
    identity["git_commit_reviewed"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    _write(output / "resolved_config.json", {"identity": identity, "config": bundle["resolved_config"]})
    for name in (
        "architecture_audit", "observation_field_map", "observation_recoverability",
        "observation_aliasing", "opportunity_identifiability", "information_gain",
        "marl_necessity_verdict", "input_validation_report",
    ):
        _write(output / f"{name}.json", {"identity": identity, "report": bundle[name]})
    _write(output / "synthetic_validation.json", {"identity": identity, "report": build_synthetic_validation_report()})
    command = [sys.executable, *sys.argv]
    command_log = {
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": command,
        "shell_command": shlex.join(command),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "information_sufficiency_audit_contract_version": INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION,
        "reward_read": False,
        "aggregate_read": False,
        "oracle_future_treated_as_observable": False,
        "hidden_read": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "formal_holdout_hidden_executed": False,
        "g11_executed": False,
        "input_paths": {
            "fairness_manifest": str(Path(args.fairness_manifest_path).resolve()),
            "request_replay": str(Path(args.request_replay_path).resolve()),
            "oracle_action_trace": str(Path(args.oracle_action_trace_path).resolve()),
            "opportunity_rows": str(Path(args.opportunity_rows_path).resolve()),
            "observation_trace": str(Path(args.observation_trace_path).resolve()) if args.observation_trace_path else None,
        },
    }
    _write(output / "command_log.json", command_log)
    files = sorted(path for path in output.iterdir() if path.is_file())
    integrity = {
        "artifact_integrity_manifest_version": "1.0.0",
        "information_sufficiency_audit_contract_version": INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION,
        "audit_fingerprint": identity["audit_fingerprint"],
        "manifest_validation_status": manifest_report["status"],
        "request_replay_validation_status": replay_report["status"],
        "audit_input_validation_status": bundle["input_validation_report"]["status"],
        "files": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }
    _write(output / "artifact_integrity_manifest.json", integrity)
    print(json.dumps({
        "output_dir": str(output),
        "audit_fingerprint": identity["audit_fingerprint"],
        "architecture": bundle["architecture_audit"]["current_architecture_classification"],
        "entity_level_marl_verdict": bundle["marl_necessity_verdict"]["entity_level_marl_evidence"],
        "overall_verdict": bundle["marl_necessity_verdict"]["overall_verdict"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
