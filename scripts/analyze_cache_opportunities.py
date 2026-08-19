"""Build a G09 cache opportunity analyzer artifact from matched raw inputs."""

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

from src.analysis.cache_opportunity_analyzer import (  # noqa: E402
    CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION,
    analyze_cache_opportunities,
)
from src.evaluators.cache_baseline_fairness import load_and_validate_manifest  # noqa: E402
from src.oracles.cache_request_replay import load_and_validate_request_replay  # noqa: E402
from src.oracles.future_horizon_cache_oracle import build_observed_baseline_outcome  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Analyze matched cache placement opportunities")
    parser.add_argument("--fairness_manifest_path", required=True)
    parser.add_argument("--request_replay_path", required=True)
    parser.add_argument("--oracle_results_path", required=True)
    parser.add_argument("--oracle_action_trace_path", required=True)
    parser.add_argument("--baseline_outcome_path", nargs="+", required=True, help="Raw CacheEvent episode summaries or observed outcomes containing request_outcomes")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--analyzer_config_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analyzer artifact directory: {output}")
    manifest, manifest_report = load_and_validate_manifest(args.fairness_manifest_path, root=ROOT)
    replay, replay_report = load_and_validate_request_replay(args.request_replay_path, source_manifest=manifest)
    oracle_results = _load(args.oracle_results_path)
    oracle_traces = _load(args.oracle_action_trace_path)
    outcomes: dict[str, Any] = {}
    for path in args.baseline_outcome_path:
        value = _load(path)
        if "cache_event_trace" in value:
            value = build_observed_baseline_outcome(replay=replay, manifest=manifest, summary=value)
        identity = str(value.get("baseline_identity"))
        if identity in outcomes:
            raise ValueError(f"duplicate baseline identity: {identity}")
        outcomes[identity] = value
    config = _load(args.analyzer_config_path) if args.analyzer_config_path else None
    bundle = analyze_cache_opportunities(
        manifest=manifest, replay=replay, oracle_results=oracle_results,
        oracle_action_traces=oracle_traces, baseline_outcomes=outcomes,
        horizons=args.horizons, config=config,
    )
    output.mkdir(parents=True)
    provenance = bundle["opportunity_summary"]["identity"]
    for name, value in bundle.items():
        if name == "opportunity_summary":
            payload = value
        elif isinstance(value, list):
            payload = {"provenance": provenance, "rows": value}
        else:
            payload = {"provenance": provenance, "report": value}
        _write(output / f"{name}.json", payload)
    command_log = {
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cache_opportunity_analyzer_contract_version": CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION,
        "reward_read": False, "legacy_aggregate_read": False, "learned_policy_hidden_state_read": False,
        "hidden_read": False, "training_executed": False, "formal_benchmark_executed": False,
        "input_paths": {
            "fairness_manifest": str(Path(args.fairness_manifest_path).resolve()),
            "request_replay": str(Path(args.request_replay_path).resolve()),
            "oracle_results": str(Path(args.oracle_results_path).resolve()),
            "oracle_action_trace": str(Path(args.oracle_action_trace_path).resolve()),
            "baseline_outcomes": [str(Path(path).resolve()) for path in args.baseline_outcome_path],
        },
    }
    _write(output / "command_log.json", command_log)
    files = sorted(path for path in output.iterdir() if path.is_file())
    integrity = {
        "artifact_integrity_manifest_version": "1.0.0",
        "analyzer_contract_version": CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION,
        "manifest_validation_status": manifest_report["status"],
        "request_replay_validation_status": replay_report["status"],
        "analyzer_input_validation_status": bundle["input_validation_report"]["status"],
        "reconciliation_status": bundle["reconciliation_report"]["status"],
        "files": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }
    _write(output / "artifact_integrity_manifest.json", integrity)
    print(json.dumps({
        "output_dir": str(output),
        "analysis_fingerprint": bundle["opportunity_summary"]["identity"]["analysis_fingerprint"],
        "request_analysis_rows": len(bundle["request_opportunity_rows"]),
        "reconciliation_status": bundle["reconciliation_report"]["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
