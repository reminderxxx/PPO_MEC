"""Build the create-only G14R20-I5 acceptance and unsigned-request package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ORIGINAL_PROJECT_GRANT_PATH,
    ORIGINAL_REQUEST_PATH,
    ORIGINAL_RUN_ROOT,
    build_restricted_recovery_request,
    validate_restricted_recovery_request,
)


PROTECTED_USER_FILE_START_SHA256 = {
    "scripts/train_sa_ghmappo_real_sample.py": "4381aa5bdddb56e321dd2a72d8c51c12e9aba57d28f83afae72e9985ff13a992",
    "src/agents/sa_ghmappo_agent.py": "31abd07c544e9eef94349483cff5335e763750670ab570e860e0a6545ae743be",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recovery-execution-id", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--executor-checkout", required=True)
    parser.add_argument("--executor-commit", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--targeted-junit", required=True)
    parser.add_argument("--adjacent-junit", required=True)
    parser.add_argument("--full-junit")
    return parser


def write_create_only(path: Path, payload: object) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def junit_summary(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": sum(int(suite.get("tests", "0")) for suite in suites),
        "failures": sum(int(suite.get("failures", "0")) for suite in suites),
        "errors": sum(int(suite.get("errors", "0")) for suite in suites),
        "skipped": sum(int(suite.get("skipped", "0")) for suite in suites),
        "time_seconds": sum(float(suite.get("time", "0")) for suite in suites),
    }


def protected_user_files() -> dict[str, object]:
    rows = []
    for relative, expected in PROTECTED_USER_FILE_START_SHA256.items():
        path = Path("/Users/howen/Projects/PPO_MEC") / relative
        observed = file_sha256(path)
        # Every hash comes from the task-start snapshot, preventing the
        # acceptance package from silently blessing concurrent changes.
        rows.append(
            {
                "path": str(path),
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "inode": path.stat().st_ino,
                "sha256": observed,
                "task_start_sha256": expected,
                "unchanged_from_task_start": expected == observed if expected else None,
            }
        )
    return {
        "status": "pass" if all(row["unchanged_from_task_start"] is not False for row in rows) else "fail",
        "files": rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_root)
    if output.exists():
        raise ValueError("acceptance output root must not already exist")
    output.mkdir(parents=True)
    request = build_restricted_recovery_request(
        original_request_path=ORIGINAL_REQUEST_PATH,
        original_project_grant_path=ORIGINAL_PROJECT_GRANT_PATH,
        original_run_root=ORIGINAL_RUN_ROOT,
        recovery_execution_id=args.recovery_execution_id,
        recovery_root=args.recovery_root,
        executor_checkout=args.executor_checkout,
        executor_commit=args.executor_commit,
        python_executable=args.python_executable,
    )
    validate_restricted_recovery_request(request, check_live=False)
    write_create_only(output / "authorization_request_unsigned.json", request)
    write_create_only(output / "readonly_source_qualification.json", request["immutable_source_audit"])
    write_create_only(output / "scientific_invariance.json", request["scientific_invariance"])
    write_create_only(
        output / "identity_bridge.json",
        {
            "original_identity": request["original_identity"],
            "recovery_execution_identity": request["recovery_execution"],
            "identities_are_equal": False,
            "bridge_sha256": canonical_sha256(
                {
                    "original": request["original_identity"],
                    "recovery": request["recovery_execution"][
                        "recovery_execution_identity_sha256"
                    ],
                }
            ),
        },
    )
    write_create_only(
        output / "producer_consumer_matrix.json",
        {
            "version": "1.0.0",
            "external_original_cells": request["external_committed_cells"],
            "planned_recovery_cells": request["recovery_execution"]["cell_specs"],
            "deduplication": request["source_mapping_and_deduplication"],
            "statistics_consumer": {
                "input": "future ablation_recovery_handoff.json",
                "formal_controller_source": "three external original committed cells",
                "direct_legacy_single_root_consumption": False,
                "explicit_path_resolver_available_after_ablation_handoff": True,
                "executed_in_this_acceptance": False,
                "separate_authorization_required": True,
            },
        },
    )
    execution = request["recovery_execution"]
    commands = {
        "version": "1.0.0",
        "working_directory": args.executor_checkout,
        "prepare_and_validate_authorized_now": True,
        "execute_authorized_now": False,
        "prepare": [
            args.python_executable,
            str(ROOT / "scripts/prepare_typed_model_cache_restricted_recovery.py"),
            "--action",
            "prepare",
            "--original-request-path",
            str(ORIGINAL_REQUEST_PATH),
            "--original-project-grant-path",
            str(ORIGINAL_PROJECT_GRANT_PATH),
            "--original-run-root",
            str(ORIGINAL_RUN_ROOT),
            "--recovery-execution-id",
            args.recovery_execution_id,
            "--recovery-root",
            args.recovery_root,
            "--executor-checkout",
            args.executor_checkout,
            "--executor-commit",
            args.executor_commit,
            "--python-executable",
            args.python_executable,
            "--output-path",
            str(output / "authorization_request_unsigned.json"),
        ],
        "validate": [
            args.python_executable,
            str(ROOT / "scripts/prepare_typed_model_cache_restricted_recovery.py"),
            "--action",
            "validate",
            "--output-path",
            str(output / "authorization_request_unsigned.json"),
        ],
        "future_execute_plan_requires_new_grant": [
            [
                args.python_executable,
                str(ROOT / "scripts/run_typed_model_cache_restricted_recovery.py"),
                "--authorization-request-path",
                str(output / "authorization_request_unsigned.json"),
                "--recovery-grant-path",
                "<not_issued_exact_recovery_grant.json>",
                "--cell-id",
                cell_id,
                "--check",
                "execute",
            ]
            for cell_id in execution["allowed_cell_ids"]
        ],
        "scientific_commands": execution["command_plan"]["commands"],
        "automatic_next_phase": False,
    }
    write_create_only(output / "complete_command_plan.json", commands)

    junit_inputs = {
        "targeted": Path(args.targeted_junit),
        "adjacent": Path(args.adjacent_junit),
    }
    if args.full_junit:
        junit_inputs["full"] = Path(args.full_junit)
    test_results = {}
    for label, path in junit_inputs.items():
        if not path.is_file():
            raise ValueError(f"missing JUnit evidence: {path}")
        destination = output / f"{label}_tests.junit.xml"
        shutil.copyfile(path, destination)
        test_results[label] = {
            **junit_summary(destination),
            "path": destination.name,
            "sha256": file_sha256(destination),
        }
    public_summary_source = Path(args.targeted_junit).parent / "public_recovery_acceptance.json"
    if not public_summary_source.is_file():
        raise ValueError("public acceptance counter evidence is missing")
    shutil.copyfile(public_summary_source, output / "public_recovery_acceptance.json")
    public_summary = json.loads((output / "public_recovery_acceptance.json").read_text())
    if (
        public_summary.get("negative_dispatch_count") != 0
        or public_summary.get("negative_recovery_write_count") != 0
        or public_summary.get("synthetic_dispatch_count") != 2
        or public_summary.get("successful_cell_process_count") != 2
    ):
        raise ValueError("public acceptance dispatch/write counters failed")

    protected = protected_user_files()
    if protected["status"] != "pass" or any(
        row["unchanged_from_task_start"] is None for row in protected["files"]
    ):
        raise ValueError("protected user-file task-start hashes are incomplete or changed")
    write_create_only(output / "protected_workspace_results.json", protected)
    head = subprocess.check_output(
        ["git", "-C", args.executor_checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", args.executor_checkout, "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git", "-C", args.executor_checkout, "status", "--porcelain",
            "--untracked-files=all", "--", "README.md", "docs", "scripts", "src", "tests", "configs",
        ],
        text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).strip()
    acceptance = {
        "status": "READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION",
        "reviewed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "literature_cutoff": "2026-06-21",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": "g14r20_i5_restricted_recovery_acceptance_20260914",
        "policy_version": "tmc_review_policy_v3_20260621",
        "evidence_level": "E2_ARTIFACT_AUDITED_SYNTHETIC_RECOVERY_EXECUTION",
        "paper_verdict": "Unverifiable",
        "base_commit": "4e2dadda644e803f9d4f3a279a46db9288d00094",
        "base_tree": "987074bb7cbf3de569229313eb44a2be3af39e2f",
        "executor_commit": head,
        "executor_git_tree": tree,
        "executor_matches_request": head == args.executor_commit == execution["executor_commit"],
        "clean_code_scope": not bool(status),
        "test_results": test_results,
        "public_acceptance": public_summary,
        "source_qualification": "pass_retention_only",
        "scientific_parameters_unchanged": True,
        "formal_performance_conclusion": False,
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    if not acceptance["executor_matches_request"] or not acceptance["clean_code_scope"]:
        raise ValueError("final executor commit/tree/clean-scope check failed")
    write_create_only(output / "acceptance_summary.json", acceptance)
    manifest_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "integrity_manifest.json":
            manifest_rows.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "version": "1.0.0",
        "artifact_run_id": acceptance["artifact_run_id"],
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_create_only(output / "integrity_manifest.json", manifest)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
