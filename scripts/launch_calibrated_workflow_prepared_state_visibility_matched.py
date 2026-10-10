"""Launch the one-shot prepared-state visibility matched development run."""

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

from scripts.analyze_calibrated_workflow_prepared_state_visibility_matched import analyze  # noqa: E402
from scripts.launch_calibrated_workflow_value_normalization_ab import (  # noqa: E402
    DEFAULT_INTERPRETER,
    _dispatch,
    _frozen_interpreter,
    _sha256,
    _verify,
    _write_json,
)
from scripts.run_calibrated_workflow_prepared_state_visibility_matched import (  # noqa: E402
    DEFAULT_HISTORICAL_ROOT,
    DEFAULT_PROTOCOL,
    DEFAULT_RULE_ROOT,
)


RUN_ID = "cscwd_causal_prepared_state_visibility_matched_20261010_v1"
RUN_ROOT = ROOT_DIR / "artifacts/experiments" / RUN_ID
ANALYSIS_ROOT = ROOT_DIR / "artifacts/analysis" / f"{RUN_ID}_analysis_v1"
SUPERVISOR_ROOT = ROOT_DIR / "artifacts/experiments" / f"{RUN_ID}_supervisor"


def _job(expected_commit: str, historical_root: Path, rule_root: Path) -> int:
    command = [
        DEFAULT_INTERPRETER,
        str(ROOT_DIR / "scripts/run_calibrated_workflow_prepared_state_visibility_matched.py"),
        "--run", "--protocol", DEFAULT_PROTOCOL,
        "--expected-git-commit", expected_commit,
        "--output-root", str(RUN_ROOT),
        "--historical-root", str(historical_root),
        "--rule-root", str(rule_root),
    ]
    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if result.returncode:
        return result.returncode
    try:
        analyze(RUN_ROOT, ANALYSIS_ROOT)
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
    parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    parser.add_argument("--rule-root", default=str(DEFAULT_RULE_ROOT))
    args = parser.parse_args()
    historical_root = Path(args.historical_root).resolve()
    rule_root = Path(args.rule_root).resolve()
    if args.mode == "verify":
        return _verify(SUPERVISOR_ROOT, 0.0, True)
    if not args.expected_commit:
        raise RuntimeError("frozen commit is required")
    if args.mode == "job":
        return _job(args.expected_commit, historical_root, rule_root)
    interpreter = _frozen_interpreter(DEFAULT_INTERPRETER)
    plan = {
        "schema_version": "persistent_ab_supervisor_plan_v1", "kind": "scientific",
        "run_id": RUN_ID, "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_commit": args.expected_commit, "interpreter": interpreter,
        "cwd": str(ROOT_DIR), "supervisor_root": str(SUPERVISOR_ROOT),
        "output_root": str(RUN_ROOT), "authorization_config": DEFAULT_PROTOCOL,
        "authorization_sha256": _sha256(ROOT_DIR / DEFAULT_PROTOCOL),
        "historical_root": str(historical_root),
        "historical_evaluation_rows_sha256": _sha256(historical_root / "evaluation_rows.csv"),
        "rule_root": str(rule_root), "rule_evaluation_rows_sha256": _sha256(rule_root / "evaluation_rows.csv"),
        "child_argv": [interpreter, str(Path(__file__).resolve()), "--mode", "job",
                       "--expected-commit", args.expected_commit,
                       "--historical-root", str(historical_root), "--rule-root", str(rule_root)],
        "entry_receipt": str(RUN_ROOT / "runner_entered.json"),
        "expected_exit_code": 0, "entry_timeout_seconds": 30.0,
        "wall_clock_cap_seconds": 7200.0, "automatic_retry": False,
        "formal_or_holdout_reads": 0,
    }
    ack = _dispatch(plan)
    print(json.dumps({
        "status": "entered_child", "ack": ack,
        "stdout": str(SUPERVISOR_ROOT / "child_stdout.log"),
        "stderr": str(SUPERVISOR_ROOT / "child_stderr.log"),
        "terminal": str(SUPERVISOR_ROOT / "terminal_receipt.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
