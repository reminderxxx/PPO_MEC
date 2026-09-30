"""Finalize integrity, command, data-binding, and workspace-protection records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROTECTED_FILES = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}
PROTECTED_DIFF_SHA256 = "1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def protection(main_root: Path) -> dict[str, Any]:
    actual = {path: sha256_file(main_root / path) for path in PROTECTED_FILES}
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *PROTECTED_FILES], cwd=main_root
    )
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=main_root, text=True
    ).splitlines()
    status_paths = [line[3:] for line in status]
    passed = bool(
        actual == PROTECTED_FILES
        and sha256_bytes(diff) == PROTECTED_DIFF_SHA256
        and len(status) == 7
        and sorted(status_paths) == sorted(PROTECTED_FILES)
    )
    return {
        "main_workspace": str(main_root),
        "start_snapshot_source": "native_typed_cache_replacement_20260930_v1/main_workspace_seven_file_protection.json",
        "expected_and_start_file_sha256": PROTECTED_FILES,
        "end_file_sha256": actual,
        "expected_and_start_combined_binary_diff_sha256": PROTECTED_DIFF_SHA256,
        "end_combined_binary_diff_sha256": sha256_bytes(diff),
        "end_git_status_short": status,
        "exactly_seven_protected_files": len(status) == 7,
        "start_end_match": passed,
        "protection_pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    protect = protection(args.main_root)
    if not protect["protection_pass"]:
        raise RuntimeError("main workspace protection snapshot drifted")
    write(output / "main_workspace_protection.json", protect)

    execution = {
        "artifact_run_id": "remaining_workflow_decision_value_audit_20260930_v1",
        "fixed_start_commit": "58c3a8900152f03e0d655194e1a02afb50f1f727",
        "previous_diagnostic_execution_commit": "ffa10e598feddac99c2557d994ba094692ede99a",
        "branch": git(args.worktree_root, "branch", "--show-current"),
        "head_at_execution": git(args.worktree_root, "rev-parse", "HEAD"),
        "worktree": str(args.worktree_root),
        "data_binding": {
            "method": "temporary_symlink_to_identity-matched_existing_main-workspace_files",
            "restored_after_test": True,
            "downloads": 0,
            "copies": 0,
            "mocks": 0,
            "alibaba": {
                "pointer_bytes": 134,
                "pointer_sha256": "14d63f5b68fcba83931563b8bff35d6e22df043b470604e233b1811c04027a3f",
                "lfs_oid_and_real_sha256": "6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc",
                "real_bytes": 802261444,
            },
            "ngsim": {
                "pointer_bytes": 135,
                "pointer_sha256": "48c89cbaccd30352e25d7f29e2f2029cd91e95d84aa0821ba1399fdc00e0d054",
                "lfs_oid_and_real_sha256": "ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c",
                "real_bytes": 2118175938,
            },
        },
        "commands": [
            {
                "command": "/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q tests/test_typed_runtime_plumbing.py",
                "result": "34 passed in 7.07s",
                "data_binding_active": True,
            },
            {
                "command": "python scripts/audit_remaining_workflow_decision_value.py --old-log <old> --candidate-log <candidate> --output-root <run>",
                "result": "8 ledger rows; 4 equal-completion candidate rows; migration modes keep/prepare",
            },
            {
                "command": "driving_pilot_py39_v1/bin/python scripts/measure_local_model_and_state_costs.py --plan <plan> --output <raw>",
                "result": "3/3 base loads, 3/3 inference, 3/3 state restore passed",
            },
            {
                "command": "python scripts/run_remaining_workflow_paired_witness.py --design <design> --output <raw>",
                "result": "16 episodes exactly once; 6 decision changes; two-step matched full-tail",
            },
            {
                "command": "python -m pytest -q tests/test_remaining_workflow_decision_value.py tests/test_typed_cache_sequential_recompute.py tests/test_typed_model_cache.py tests/test_typed_model_cache_runtime.py tests/test_cache_event_contract.py tests/test_cache_capacity_mb.py tests/test_cache_eviction_policy.py tests/test_env_contract.py",
                "result": "80 passed in 1.01s",
            },
            {
                "command": "python scripts/smoke_test.py",
                "result": "6 nodes completed; terminated=True; truncated=False",
            },
            {
                "command": "python -m compileall -q <four scripts> <test> && git diff --check",
                "result": "pass",
            },
        ],
        "bounded_execution_budget": {
            "paired_episode_limit": 24,
            "paired_episodes_executed": 16,
            "paired_scientific_reruns": 0,
            "model_load_warmups": 1,
            "model_load_measurements": 3,
            "inference_warmups_per_measured_process": 1,
            "inference_measurements_per_measured_process": 1,
            "state_warmups": 1,
            "state_measurements": 3,
            "training_runs": 0,
            "formal_runs": 0,
            "holdout_runs": 0,
        },
        "failure_records": [
            {
                "stage": "ledger consumer development before paired/model execution",
                "error": "TypeError: ordered_object_ids was initially read as object dictionaries",
                "disposition": "minimal schema correction; no scientific episode or measurement consumed",
            },
            {
                "stage": "ledger consumer development before paired/model execution",
                "error": "KeyError: resident_object_ids; original field is native_object_ids",
                "disposition": "minimal schema correction; no scientific episode or measurement consumed",
            },
            {
                "stage": "artifact finalizer after all scientific execution",
                "error": "workspace status parser stripped the leading porcelain column",
                "disposition": "parser-only correction; protected hashes and diff never changed",
            },
        ],
        "scope_guards": {
            "no_training": True,
            "no_dataset_expansion": True,
            "no_formal_or_holdout": True,
            "no_model_reselection": True,
            "no_download": True,
            "no_os_cache_clear": True,
            "no_public_upload": True,
        },
    }
    write(output / "commands_environment_failures.json", execution)

    files = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "artifact_integrity_manifest.json":
            continue
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write(
        output / "artifact_integrity_manifest.json",
        {
            "artifact_run_id": "remaining_workflow_decision_value_audit_20260930_v1",
            "status": "pass",
            "file_count": len(files),
            "files": files,
        },
    )
    print(json.dumps({"file_count": len(files), "protection_pass": True}, sort_keys=True))


if __name__ == "__main__":
    main()
