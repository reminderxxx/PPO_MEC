#!/usr/bin/env python3
"""Run one frozen G14R21 statistics rebuild with durable local receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    output_root = args.output_root.resolve()
    job_root = output_root.parent / "local_job"
    state_path = job_root / "state.json"
    receipt_path = job_root / "completion_receipt.json"
    stdout_path = job_root / "stdout.log"
    stderr_path = job_root / "stderr.log"
    old = json.loads((formal_root / "statistics/paired_statistics.json").read_text(encoding="utf-8-sig"))
    metrics = list(dict.fromkeys(row["metric"] for row in old["rows"]))
    identity = old["pairwise_comparison_identity"]
    command = [sys.executable, str(ROOT / "scripts/analyze_top_journal_statistics.py")]
    for path in old["source_rows_path"]:
        command.extend(["--rows_path", path])
    command.extend(
        [
            "--candidate_agent", identity["candidate_agent"],
            "--baseline_agents", *identity["baseline_agent_order"],
            "--metrics", *metrics,
            "--pair_keys", *old["pair_keys"],
            "--outer_cluster_keys", *old["outer_cluster_keys"],
            "--inner_cluster_keys", *old["inner_cluster_keys"],
            "--ci_method", old["requested_ci_method"],
            "--bootstrap_samples", str(old["bootstrap_samples"]),
            "--random_seed", "1401",
            "--formal-agent-order-contract-path",
            str(ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/formal_agent_order_contract.json"),
            "--output_root", str(output_root),
        ]
    )
    job_root.mkdir(parents=True, exist_ok=True)
    started = now()
    state = {
        "job_contract_version": "g14r21_local_statistics_job_v1",
        "status": "running",
        "supervisor_pid": os.getpid(),
        "started_at": started,
        "command": command,
        "cwd": str(ROOT),
        "allowed_write_roots": [str(output_root), str(job_root)],
        "automatic_retry_count": 0,
    }
    write_json(state_path, state)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=stdout, stderr=stderr, check=False)
    completed = now()
    validation_error = None
    if result.returncode == 0:
        try:
            payload = json.loads((output_root / "paired_statistics.json").read_text(encoding="utf-8"))
            if len(payload.get("rows", [])) != 84:
                raise ValueError("corrected statistics does not contain exactly 84 comparisons")
            if any(row.get("holm_preregistered_family_size") != 84 for row in payload["rows"]):
                raise ValueError("Holm family identity mismatch")
        except Exception as exc:  # recorded as a terminal validation failure
            validation_error = str(exc)
    success = result.returncode == 0 and validation_error is None
    receipt = {
        **state,
        "status": "completed" if success else "failed",
        "completed_at": completed,
        "return_code": result.returncode,
        "artifact_validation_passed": validation_error is None and result.returncode == 0,
        "artifact_validation_error": validation_error,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_path": str(output_root / "paired_statistics.json"),
    }
    write_json(receipt_path, receipt)
    write_json(state_path, receipt)
    return 0 if success else (result.returncode or 2)


if __name__ == "__main__":
    raise SystemExit(main())
