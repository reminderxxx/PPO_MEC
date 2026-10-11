"""Durably launch the one-shot positive-prepare calibration-abstention A/B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cscwd_positive_prepare_calibration_abstention_ab import (  # noqa: E402
    analyze,
)
from scripts.launch_calibrated_workflow_value_normalization_ab import (  # noqa: E402
    DEFAULT_INTERPRETER,
    SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
    _dispatch,
    _frozen_interpreter,
    _sha256,
    _verify,
    _write_json,
)


PROTOCOL = (
    "configs/experiment/"
    "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_v1.json"
)
RUN_ID = "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_20261011_v1"
RUN_ROOT = ROOT / "artifacts/experiments" / RUN_ID
ANALYSIS_ROOT = ROOT / "artifacts/analysis" / f"{RUN_ID}_analysis_v1"
SUPERVISOR_ROOT = ROOT / "artifacts/experiments" / f"{RUN_ID}_supervisor"
REQUIRED_REMOTE_REF = "origin/codex/event-aux-abstention"


def _remote_contains(expected_commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_commit, REQUIRED_REMOTE_REF],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _job(expected_commit: str) -> int:
    command = [
        DEFAULT_INTERPRETER,
        str(ROOT / "scripts/run_cscwd_phase_v2_causal_public_pilot.py"),
        "--run",
        "--protocol",
        PROTOCOL,
        "--expected-git-commit",
        expected_commit,
        "--output-root",
        str(RUN_ROOT),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        return result.returncode
    try:
        analyze(RUN_ROOT, ANALYSIS_ROOT)
    except Exception as error:
        _write_json(RUN_ROOT / "analysis_failure_receipt.json", {
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        })
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "job", "verify"), required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.mode == "verify":
        return _verify(SUPERVISOR_ROOT, 0.0, True)
    if not args.expected_commit:
        raise RuntimeError("frozen commit is required")
    if args.mode == "job":
        return _job(args.expected_commit)
    if not _remote_contains(args.expected_commit):
        raise RuntimeError(
            f"frozen commit is not present on {REQUIRED_REMOTE_REF}; launch refused"
        )
    interpreter = _frozen_interpreter(DEFAULT_INTERPRETER)
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("execution_authorized") is not True:
        raise RuntimeError("scientific execution is not literally authorized")
    if protocol.get("run_id") != RUN_ID:
        raise RuntimeError("scientific run identity drift")
    plan = {
        "schema_version": "persistent_ab_supervisor_plan_v1",
        "kind": "scientific",
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_commit": args.expected_commit,
        "interpreter": interpreter,
        "cwd": str(ROOT),
        "supervisor_root": str(SUPERVISOR_ROOT),
        "output_root": str(RUN_ROOT),
        "authorization_config": PROTOCOL,
        "authorization_sha256": _sha256(ROOT / PROTOCOL),
        "required_remote_ref": REQUIRED_REMOTE_REF,
        "remote_commit_verified_before_dispatch": True,
        "child_argv": [
            interpreter,
            str(Path(__file__).resolve()),
            "--mode",
            "job",
            "--expected-commit",
            args.expected_commit,
        ],
        "entry_receipt": str(RUN_ROOT / "runner_entered.json"),
        "expected_exit_code": 0,
        "entry_timeout_seconds": 60.0,
        "scientific_completion_receipt_schema_version": (
            SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION
        ),
        "wall_clock_cap_seconds": 3600.0,
        "automatic_retry": False,
        "formal_or_holdout_reads": 0,
    }
    ack = _dispatch(plan)
    print(json.dumps({
        "status": "entered_child",
        "ack": ack,
        "stdout": str(SUPERVISOR_ROOT / "child_stdout.log"),
        "stderr": str(SUPERVISOR_ROOT / "child_stderr.log"),
        "terminal": str(SUPERVISOR_ROOT / "terminal_receipt.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
