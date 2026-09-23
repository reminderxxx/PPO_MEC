#!/usr/bin/env python3
"""Create the G14R22 validation summary and self-excluding integrity manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.dedicated_holdout_execution import canonical_sha256, file_sha256


def write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-summary", required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    request = json.loads((root / "holdout_request_unsigned.json").read_text())
    package = json.loads((root / "command_package.json").read_text())
    acceptance = json.loads((root / "non_holdout_acceptance/acceptance_receipt.json").read_text())
    checkpoint = json.loads((root / "checkpoint_byte_audit_preopen.json").read_text())
    summary = {
        "analysis_summary_version": "g14r22_v1",
        "verdict": "READY_TO_REQUEST_HOLDOUT_EXECUTION_AUTHORIZATION",
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "executor_commit": package["executor_commit"],
        "checkpoint_models_rehashed": checkpoint["actual_scope"]["models"],
        "checkpoint_bytes_rehashed": checkpoint["actual_scope"]["total_bytes"],
        "non_holdout_public_acceptance_passed": acceptance["passed"],
        "non_holdout_scientific_children": acceptance["scientific_children"],
        "statistics_rows": acceptance["statistics_rows"],
        "test_summary": args.test_summary,
        "grant_signed": False,
        "execution_authorized": False,
        "holdout_opened": False,
        "holdout_policy_runs": 0,
        "tested_full_run_upper_bound_seconds": None,
    }
    write(root / "analysis_summary.json", summary)
    excluded = {"artifact_integrity_manifest.json"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root).as_posix() in excluded:
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    manifest = {
        "artifact_integrity_version": "g14r22_v1",
        "file_count_excluding_manifest": len(files),
        "files": files,
        "files_canonical_sha256": canonical_sha256(files),
        "self_excluded": ["artifact_integrity_manifest.json"],
    }
    write(root / "artifact_integrity_manifest.json", manifest)
    print(json.dumps({"status": "pass", "file_count": len(files),
                      "files_canonical_sha256": manifest["files_canonical_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
