"""Validate a detached G14R11 candidate without formal execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_formal_request_subject_lifecycle_rehearsal as rehearsal
from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.formal_window_consumption import load_contract, validate_reachability
from src.evaluators.typed_model_cache_formal_execution import (
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.runtime.active_formal_bundle import (
    build_active_bundle_resource_resolution_audit,
    validate_active_formal_bundle,
)


PROTOCOL_ROOT = ROOT / (
    "configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901"
)
WINDOW_CONTRACT = ROOT / (
    "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821/"
    "formal_window_consumption_contract.json"
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--shared-data-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    shared_data_root = Path(args.shared_data_root).resolve()
    workflow_csv = shared_data_root / "raw/workflow/alibaba2018/batch_task.csv"
    if not workflow_csv.is_file():
        raise FileNotFoundError(workflow_csv)
    candidate_commit = git("rev-parse", "HEAD")
    if candidate_commit != args.candidate_commit:
        raise RuntimeError("detached candidate commit mismatch")
    detached_head = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).returncode != 0
    git_status = git("status", "--porcelain")
    local_venv_present = (ROOT / ".venv").exists()
    if not detached_head or git_status or local_venv_present:
        raise RuntimeError("candidate must be detached, Git-clean, and without local .venv")

    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_ready=args.require_ready,
        require_clean_git=True,
        require_origin_main_match=False,
    )
    protocol_report = validate_protocol_v1_1(bundle["protocol"])
    resource_audit = build_active_bundle_resource_resolution_audit(bundle)

    work_root = Path(args.work_root).resolve()
    expansion = resolved_expansion_context(
        bundle["protocol"],
        protocol_path=bundle["protocol_path"],
        output_root=str(work_root / "formal-output-not-created"),
        python_executable=sys.executable,
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
        active_protocol_index_path=str(PROTOCOL_ROOT / "protocol_index.json"),
        active_bundle_resource_resolution_audit_sha256=resource_audit[
            "audit_sha256"
        ],
    )
    command_report = validate_command_templates(
        bundle["protocol"]["execution_contract"]["command_templates"], expansion
    )
    commands = [
        row
        for phase in command_report["expanded"].values()
        for row in phase["commands"]
    ]
    unresolved_placeholder_count = sum(
        "{" in token or "}" in token for row in commands for token in row
    )
    absolute_sentinel_count = sum(
        "/ABSOLUTE/" in token for row in commands for token in row
    )
    if (
        command_report["command_count"] != 186
        or unresolved_placeholder_count
        or absolute_sentinel_count
    ):
        raise RuntimeError("frozen command expansion is incomplete")

    reachability = validate_reachability(WINDOW_CONTRACT)
    if reachability["status"] != "pass" or reachability["reachable_count"] != 60:
        raise RuntimeError("60/60 frozen window reachability failed")

    rehearsal.WORKFLOW_CSV = workflow_csv
    exact_root = work_root / "exact-rehearsal"
    exact = rehearsal.exact_failure_unit_rehearsal(exact_root)
    contract = load_contract(WINDOW_CONTRACT)
    unit = next(
        row
        for row in contract["evaluation_units"]
        if row["split_name"] == "train"
        and row["window_id"] == rehearsal.FAILURE_WINDOW
    )
    workflow = rehearsal.workflows()[0]
    catalog = AdapterCatalog.from_json(rehearsal.CATALOG_PATH)
    phase_fingerprints = {}
    for phase in ("train", "dev", "formal"):
        trace, _ = rehearsal.build_trace(
            split="train",
            unit=unit,
            workflow=workflow,
            catalog=catalog,
            phase=phase,
        )
        phase_fingerprints[phase] = trace["request_exposure_fingerprint"]
    if len(set(phase_fingerprints.values())) != 1:
        raise RuntimeError("train/dev/formal producer semantics diverge")

    checkpoint_files = sorted(
        path.as_posix()
        for path in work_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    )
    payload = {
        "clean_candidate_validation_version": "1.0.0",
        "status": "pass",
        "candidate_commit": candidate_commit,
        "detached_head": detached_head,
        "git_clean": git_status == "",
        "local_venv_present": local_venv_present,
        "python_executable": str(Path(sys.executable).resolve()),
        "active_bundle_atomic_status": bundle["status"],
        "active_bundle_ready_required": args.require_ready,
        "active_bundle_core_sha256": bundle["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": bundle["active_formal_bundle_sha256"],
        "protocol_version": protocol_report["protocol_version"],
        "resource_resolution_status": resource_audit["validation_status"],
        "window_reachability_status": reachability["status"],
        "reachable_window_count": reachability["reachable_count"],
        "window_count": reachability["window_count"],
        "holdout_metadata_only": reachability["holdout_metadata_only"],
        "holdout_capability": bundle["holdout_capability"],
        "holdout_sealed": bundle["index"]["holdout_seal"]["sealed"],
        "holdout_opened": bundle["index"]["holdout_seal"]["opened"],
        "command_count": command_report["command_count"],
        "phase_count": command_report["phase_count"],
        "unresolved_placeholder_count": unresolved_placeholder_count,
        "absolute_sentinel_count": absolute_sentinel_count,
        "exact_failure_unit_rehearsal_status": exact["status"],
        "exact_request_count": exact["request_count"],
        "exact_cache_event_count": exact["request_cache_event_count"],
        "exact_external_request_denominator": exact["external_request_denominator"],
        "failure_position_crossed": exact["failure_position_step_3_crossed"],
        "train_dev_formal_exposure_fingerprints": phase_fingerprints,
        "train_dev_formal_producer_semantics_identical": True,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "checkpoint_files_created": checkpoint_files,
        "g14c_v12_started": False,
        "g14d_started": False,
        "g15_started": False,
    }
    if checkpoint_files:
        raise RuntimeError("clean rehearsal created a checkpoint")
    write_json(Path(args.output_path).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
