"""Launch the frozen development run once through the persistent supervisor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_strong_baselines import analyze
from scripts.launch_calibrated_workflow_value_normalization_ab import (
    DEFAULT_INTERPRETER, _dispatch, _frozen_interpreter, _sha256, _verify, _write_json,
)

CONFIG = "configs/experiment/calibrated_workflow_strong_baselines_development_v2_prefix_only.json"
RUN_ID = "cscwd_causal_strong_baselines_dev_20261009_v1"
RUN_ROOT = ROOT_DIR / "artifacts/experiments" / RUN_ID
SUPERVISOR_ROOT = ROOT_DIR / "artifacts/experiments" / f"{RUN_ID}_supervisor"


def _job(expected_commit: str) -> int:
    command = [
        DEFAULT_INTERPRETER, str(ROOT_DIR / "scripts/run_calibrated_workflow_strong_baselines.py"),
        "--run", "--config", CONFIG, "--expected_git_commit", expected_commit,
        "--output_root", str(RUN_ROOT),
    ]
    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if result.returncode:
        return result.returncode
    try:
        analyze(RUN_ROOT)
    except Exception as error:
        _write_json(RUN_ROOT / "analysis_failure_receipt.json", {
            "status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "automatic_retry": False,
        })
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("launch", "job", "verify"), required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.mode == "verify":
        return _verify(SUPERVISOR_ROOT, 0.0, True)
    if not args.expected_commit:
        raise RuntimeError("frozen commit is required")
    if args.mode == "job":
        return _job(args.expected_commit)
    interpreter = _frozen_interpreter(DEFAULT_INTERPRETER)
    plan = {
        "schema_version": "persistent_ab_supervisor_plan_v1",
        "kind": "scientific", "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_commit": args.expected_commit,
        "interpreter": interpreter, "cwd": str(ROOT_DIR),
        "supervisor_root": str(SUPERVISOR_ROOT), "output_root": str(RUN_ROOT),
        "authorization_config": CONFIG,
        "authorization_sha256": _sha256(ROOT_DIR / CONFIG),
        "child_argv": [interpreter, str(Path(__file__).resolve()), "--mode", "job", "--expected-commit", args.expected_commit],
        "entry_receipt": str(RUN_ROOT / "runner_entered.json"),
        "expected_exit_code": 0, "entry_timeout_seconds": 30.0,
        "wall_clock_cap_seconds": 7200.0, "automatic_retry": False,
        "formal_or_holdout_reads": 0,
    }
    ack = _dispatch(plan)
    print(json.dumps({"status": "entered_child", "ack": ack,
                      "stdout": str(SUPERVISOR_ROOT / "child_stdout.log"),
                      "stderr": str(SUPERVISOR_ROOT / "child_stderr.log"),
                      "terminal": str(SUPERVISOR_ROOT / "terminal_receipt.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
