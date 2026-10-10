"""One-shot durable launcher for the SA event-auxiliary abstention experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_event_aux_abstention_ab import analyze  # noqa: E402
from scripts.launch_calibrated_workflow_value_normalization_ab import (  # noqa: E402
    DEFAULT_INTERPRETER,
    _dispatch,
    _frozen_interpreter,
    _sha256,
    _verify,
    _write_json,
)
from scripts.run_calibrated_workflow_event_aux_abstention_ab import (  # noqa: E402
    DEFAULT_CONTROL_ROOT,
    DEFAULT_PROTOCOL,
)


RUN_ID = "cscwd_event_aux_abstention_ab_20261010_v1"
RUN_ROOT = ROOT_DIR / "artifacts/experiments" / RUN_ID
ANALYSIS_ROOT = ROOT_DIR / "artifacts/analysis" / f"{RUN_ID}_analysis_v1"
SUPERVISOR_ROOT = ROOT_DIR / "artifacts/experiments" / f"{RUN_ID}_supervisor"


def _job(expected_commit: str, control_root: Path) -> int:
    command = [
        DEFAULT_INTERPRETER,
        str(ROOT_DIR / "scripts/run_calibrated_workflow_event_aux_abstention_ab.py"),
        "--run",
        "--protocol",
        DEFAULT_PROTOCOL,
        "--expected-git-commit",
        expected_commit,
        "--output-root",
        str(RUN_ROOT),
        "--control-root",
        str(control_root),
    ]
    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if result.returncode:
        return result.returncode
    try:
        analyze(RUN_ROOT, control_root, ANALYSIS_ROOT)
    except Exception as error:
        _write_json(
            RUN_ROOT / "analysis_failure_receipt.json",
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
            },
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("launch", "job", "verify"), required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--control-root", default=str(DEFAULT_CONTROL_ROOT))
    args = parser.parse_args()
    control_root = Path(args.control_root).resolve()
    if args.mode == "verify":
        return _verify(SUPERVISOR_ROOT, 0.0, True)
    if not args.expected_commit:
        raise RuntimeError("frozen commit is required")
    if args.mode == "job":
        return _job(args.expected_commit, control_root)
    interpreter = _frozen_interpreter(DEFAULT_INTERPRETER)
    plan = {
        "schema_version": "persistent_ab_supervisor_plan_v1",
        "kind": "scientific",
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_commit": args.expected_commit,
        "interpreter": interpreter,
        "cwd": str(ROOT_DIR),
        "supervisor_root": str(SUPERVISOR_ROOT),
        "output_root": str(RUN_ROOT),
        "authorization_config": DEFAULT_PROTOCOL,
        "authorization_sha256": _sha256(ROOT_DIR / DEFAULT_PROTOCOL),
        "control_root": str(control_root),
        "control_artifact_integrity_sha256": _sha256(
            control_root / "artifact_integrity.json"
        ),
        "child_argv": [
            interpreter,
            str(Path(__file__).resolve()),
            "--mode",
            "job",
            "--expected-commit",
            expected_commit,
            "--control-root",
            str(control_root),
        ],
        "entry_receipt": str(RUN_ROOT / "runner_entered.json"),
        "expected_exit_code": 0,
        "entry_timeout_seconds": 30.0,
        "wall_clock_cap_seconds": 3600.0,
        "automatic_retry": False,
        "formal_or_holdout_reads": 0,
    }
    ack = _dispatch(plan)
    print(
        json.dumps(
            {
                "status": "entered_child",
                "ack": ack,
                "stdout": str(SUPERVISOR_ROOT / "child_stdout.log"),
                "stderr": str(SUPERVISOR_ROOT / "child_stderr.log"),
                "terminal": str(SUPERVISOR_ROOT / "terminal_receipt.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
