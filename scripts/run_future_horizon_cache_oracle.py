"""Run G08 exact rolling cache oracle and emit an auditable artifact bundle."""

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

from src.evaluators.cache_baseline_fairness import load_and_validate_manifest
from src.oracles.cache_request_replay import load_and_validate_request_replay
from src.oracles.future_horizon_cache_oracle import (
    ALLOWED_HORIZONS,
    build_observed_baseline_outcome,
    compare_baseline_to_oracle,
    solve_future_horizon_cache_oracle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact future-horizon cache oracle")
    parser.add_argument("--fairness_manifest_path", required=True)
    parser.add_argument("--request_replay_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(ALLOWED_HORIZONS))
    parser.add_argument("--state_limit", type=int, default=200_000)
    parser.add_argument("--observed_baseline_path", nargs="*", default=[])
    parser.add_argument("--full_trace_diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle artifact directory: {output}")
    if any(item not in ALLOWED_HORIZONS for item in args.horizons):
        raise ValueError(f"horizons must be drawn from {ALLOWED_HORIZONS}")
    manifest, manifest_report = load_and_validate_manifest(args.fairness_manifest_path, root=ROOT)
    replay, replay_report = load_and_validate_request_replay(
        args.request_replay_path, source_manifest=manifest
    )
    output.mkdir(parents=True)
    resolved = {
        "fairness_manifest_path": str(Path(args.fairness_manifest_path).resolve()),
        "request_replay_path": str(Path(args.request_replay_path).resolve()),
        "horizons": args.horizons,
        "state_limit": args.state_limit,
        "full_trace_diagnostic": args.full_trace_diagnostic,
        "hidden_input_allowed": False,
        "formal_benchmark_executed": False,
    }
    _write(output / "request_replay.json", replay)
    _write(output / "request_replay_validation.json", replay_report)
    _write(output / "oracle_config.json", resolved)
    results: dict[str, Any] = {}
    for horizon in args.horizons:
        result = solve_future_horizon_cache_oracle(
            replay=replay,
            manifest=manifest,
            horizon=horizon,
            state_limit=args.state_limit,
        )
        results[f"h_{horizon}"] = result
    if args.full_trace_diagnostic:
        results["full_trace_diagnostic"] = solve_future_horizon_cache_oracle(
            replay=replay,
            manifest=manifest,
            horizon=max(args.horizons),
            state_limit=args.state_limit,
            full_trace_diagnostic=True,
        )
    _write(output / "oracle_results.json", {
        key: {"identity": value["identity"], "performance": value["performance"]}
        for key, value in results.items()
    })
    _write(output / "oracle_action_trace.json", {
        key: value["action_trace"] for key, value in results.items()
    })
    _write(output / "capacity_invariant_audit.json", {
        key: value["capacity_invariant_audit"] for key, value in results.items()
    })
    _write(output / "horizon_information_audit.json", {
        key: value["horizon_information_audit"] for key, value in results.items()
    })
    if args.observed_baseline_path:
        observed_by_baseline: dict[str, Any] = {}
        for observed_path in args.observed_baseline_path:
            observed = json.loads(Path(observed_path).read_text(encoding="utf-8-sig"))
            if "cache_event_trace" in observed:
                observed = build_observed_baseline_outcome(
                    replay=replay, manifest=manifest, summary=observed
                )
            identity = str(observed["baseline_identity"])
            if identity in observed_by_baseline:
                raise ValueError(f"duplicate observed baseline identity: {identity}")
            observed_by_baseline[identity] = observed
        _write(output / "observed_baseline_outcomes.json", observed_by_baseline)
        gaps = {
            key: {
                baseline: compare_baseline_to_oracle(
                    oracle_result=value, observed_baseline=observed
                )
                for baseline, observed in observed_by_baseline.items()
            }
            for key, value in results.items()
        }
    else:
        gaps = {
            key: {
                "comparable_status": "unavailable",
                "incompatibility_reasons": ["observed_baseline_outcome_not_provided"],
                "gaps": None,
                "latency_gap": {"availability": "unavailable", "value": None},
            }
            for key in results
        }
    _write(output / "baseline_oracle_gap.json", gaps)
    _write(output / "command_log.json", {
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "formal_benchmark_executed": False,
        "hidden_read": False,
    })
    files = sorted(path for path in output.iterdir() if path.is_file())
    integrity = {
        "artifact_integrity_manifest_version": "1.0.0",
        "manifest_validation_status": manifest_report["status"],
        "request_replay_validation_status": replay_report["status"],
        "files": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }
    _write(output / "artifact_integrity_manifest.json", integrity)
    print(json.dumps({
        "output_dir": str(output),
        "horizons": args.horizons,
        "optimality_status": {key: value["identity"]["optimality_status"] for key, value in results.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
