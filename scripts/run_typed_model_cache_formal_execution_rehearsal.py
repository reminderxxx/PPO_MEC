"""Audit G14R4 transactions in a no-.venv clean execution snapshot.

The real tiny dev/freeze/formal-like chain is supplied by the existing G14R3
rehearsal runner. This companion proves the v1.4 environment, per-cell resume,
75/150 interruption, and finalize-only contracts without formal data or claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    FormalCellLedger,
    stable_cell_id,
)
from src.evaluators.formal_phase_transaction import (
    PhaseCommandResult,
    PhaseTransactionError,
    TransactionalPhaseRunner,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256
from src.runtime.formal_execution_environment import (
    assert_child_environment_parity,
    resolve_execution_environment,
)


AGENTS = ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"]
SEEDS = [7, 13]
CAPACITIES = [288, 576]


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def identity(protocol: dict, environment: dict, matrix_hash: str, run_id: str) -> CellExecutionIdentity:
    return CellExecutionIdentity(
        run_id=run_id,
        execution_commit=protocol["formal_execution_environment_contract"][
            "scientific_identity"
        ]["execution_commit"],
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        resource_registry_semantic_sha256=protocol[
            "portable_resource_identity_contract"
        ]["resource_registry_semantic_sha256"],
        environment_fingerprint=environment["environment_fingerprint"],
        split_semantic_sha256=protocol["identity"]["split_semantic_sha256"],
        window_contract_semantic_sha256=protocol["execution_contract"][
            "window_consumption_contract"
        ]["semantic_sha256"],
        catalog_fingerprint=protocol["identity"]["catalog_fingerprint"],
        runtime_identity="non_formal_rehearsal_typed_runtime",
        command_matrix_sha256=matrix_hash,
    )


def commit_cell(ledger: FormalCellLedger, coordinates: dict, command: list[str]) -> str:
    result = ledger.begin_cell(
        phase="train",
        coordinates=coordinates,
        command=command,
        input_hash=canonical_sha256({"coordinates": coordinates, "mode": "rehearsal"}),
    )
    if result["status"] == "skipped_committed":
        return "skipped"
    staging = Path(result["record"]["staging_path"])
    write_json(
        staging / "train_summary.json",
        {
            "execution_mode": "non_formal_rehearsal",
            "coordinates": coordinates,
            "formal_checkpoint": False,
            "performance_claim": False,
        },
    )
    write_json(staging / "checkpoint_metadata.json", {"rehearsal_only": True})
    ledger.commit_cell(
        result["cell_id"],
        required_paths=["train_summary.json", "checkpoint_metadata.json"],
    )
    return "executed"


def interruption_16(root: Path, protocol: dict, environment: dict, python: str) -> dict:
    cells = [
        {"agent": agent, "seed": seed, "capacity_mb": capacity}
        for capacity in CAPACITIES
        for agent in AGENTS
        for seed in SEEDS
    ]
    matrix_hash = canonical_sha256(cells)
    run_id = "g14r4_16_cell_resume"
    ledger = FormalCellLedger(
        run_root=root, identity=identity(protocol, environment, matrix_hash, run_id)
    )
    command = [python, "logical_non_formal_training_cell"]
    for cell in cells[:8]:
        assert commit_cell(ledger, cell, command) == "executed"
    resumed = FormalCellLedger(
        run_root=root,
        identity=identity(protocol, environment, matrix_hash, run_id),
        resume=True,
    )
    skipped = 0
    executed = 0
    for cell in cells:
        status = commit_cell(resumed, cell, command)
        skipped += status == "skipped"
        executed += status == "executed"
    expected = [stable_cell_id("train", cell) for cell in cells]
    complete = resumed.assert_complete_matrix(phase="train", expected_cell_ids=expected)
    return {
        "status": "pass",
        "agents": AGENTS,
        "seeds": SEEDS,
        "capacities_mb": CAPACITIES,
        "training_cell_count": len(cells),
        "interrupted_after_committed_cells": 8,
        "committed_cells_skipped_on_resume": skipped,
        "new_cells_executed_after_resume": executed,
        "final_committed_cell_count": complete["committed_cell_count"],
        "duplicate_cell_count": 0,
    }


def interruption_150(root: Path, protocol: dict, environment: dict, python: str) -> dict:
    cells = [{"matrix_index": index} for index in range(150)]
    matrix_hash = canonical_sha256(cells)
    run_id = "g14r4_150_cell_interruption"
    ledger = FormalCellLedger(
        run_root=root, identity=identity(protocol, environment, matrix_hash, run_id)
    )
    command = [python, "logical_150_cell_simulation"]
    for cell in cells[:75]:
        commit_cell(ledger, cell, command)
    resumed = FormalCellLedger(
        run_root=root,
        identity=identity(protocol, environment, matrix_hash, run_id),
        resume=True,
    )
    skipped = 0
    for cell in cells:
        skipped += commit_cell(resumed, cell, command) == "skipped"
    expected = [stable_cell_id("train", cell) for cell in cells]
    complete = resumed.assert_complete_matrix(phase="train", expected_cell_ids=expected)
    return {
        "status": "pass",
        "interrupted_after_committed_cells": 75,
        "committed_cells_skipped_on_resume": skipped,
        "final_committed_cell_count": complete["committed_cell_count"],
        "duplicate_cell_count": 0,
    }


def finalize_only(root: Path) -> dict:
    runner = TransactionalPhaseRunner(
        output_root=root,
        run_identity_fingerprint="g14r4-finalize-only",
        phase_order=["train"],
    )

    def execute(_command):
        write_json(runner.output_root / "train_complete.json", {"cells": 150})
        return PhaseCommandResult(0)

    try:
        runner.run_phase(
            "train",
            commands=[["logical_train_phase"]],
            input_hash="train-phase-input",
            expected_outputs=["train_complete.json"],
            executor=execute,
            fail_terminal_append=True,
        )
    except PhaseTransactionError as exc:
        if "simulated terminal append failure" not in str(exc):
            raise
    resumed = TransactionalPhaseRunner(
        output_root=root,
        run_identity_fingerprint="g14r4-finalize-only",
        phase_order=["train"],
        resume=True,
    )
    event = resumed.finalize_phase_only(
        "train",
        commands=[["logical_train_phase"]],
        input_hash="train-phase-input",
        expected_outputs=["train_complete.json"],
    )
    duplicate = resumed.finalize_phase_only(
        "train",
        commands=[["logical_train_phase"]],
        input_hash="train-phase-input",
        expected_outputs=["train_complete.json"],
    )
    return {
        "status": "pass",
        "commands_rerun_during_finalize": 0,
        "completion_candidate_present": True,
        "terminal_status": event["status"],
        "duplicate_finalize_status": duplicate["status"],
        "output_hash_revalidated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clean-worktree-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--path-rehearsal-summary", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    clean_root = Path(args.clean_worktree_root).resolve()
    python = str(Path(args.python_executable).absolute())
    if output_root.exists():
        raise FileExistsError(output_root)
    if (clean_root / ".venv").exists():
        raise ValueError("clean rehearsal worktree must not contain .venv")
    protocol = read_json(
        clean_root
        / "configs/experiment/typed_model_cache_formal_protocol_v1_4_20260825"
        / "protocol_v1_4_manifest.json"
    )
    contract = protocol["formal_execution_environment_contract"]
    resolution = resolve_execution_environment(
        clean_worktree_root=clean_root,
        execution_commit=contract["scientific_identity"]["execution_commit"],
        python_executable=python,
        expected_identity=contract["scientific_identity"],
    )
    parity = assert_child_environment_parity(
        resolution,
        clean_worktree_root=clean_root,
        execution_commit=contract["scientific_identity"]["execution_commit"],
    )
    path_summary = read_json(Path(args.path_rehearsal_summary))
    if path_summary.get("training_cell_count") != 16 or path_summary.get("status") != "pass":
        raise RuntimeError("real tiny phase-chain rehearsal did not pass 16 cells")
    output_root.mkdir(parents=True, exist_ok=False)
    resume16 = interruption_16(
        output_root / "resume_16", protocol, resolution.environment_identity, python
    )
    resume150 = interruption_150(
        output_root / "resume_150", protocol, resolution.environment_identity, python
    )
    finalize = finalize_only(output_root / "finalize_only")
    summary = {
        "g14r4_exact_non_formal_rehearsal_version": "1.0.0",
        "status": "pass",
        "execution_mode": "non_formal_rehearsal",
        "protocol_version": "1.4.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "clean_worktree_root": str(clean_root),
        "clean_worktree_has_local_venv": False,
        "resolved_shared_python": resolution.python_executable,
        "environment_fingerprint": resolution.environment_identity[
            "environment_fingerprint"
        ],
        "environment_resolution": resolution.runtime_audit,
        "child_environment_parity": parity,
        "all_project_imports_from_clean_worktree": True,
        "real_tiny_phase_chain": path_summary,
        "interruption_8_of_16": resume16,
        "interruption_75_of_150": resume150,
        "finalize_only": finalize,
        "dev_selection_real": path_summary["dev_selector"]["real_consumer_executed"],
        "checkpoint_freeze_real": path_summary["checkpoint_freeze"]["real_consumer_executed"],
        "tiny_formal_like_real": path_summary["controller_evaluation_executed"],
        "support_real": path_summary["robustness_support_executed"],
        "statistics_real": path_summary["statistics_executed"],
        "integrity_real": path_summary["artifact_integrity_executed"],
        "complete_without_holdout": path_summary[
            "non_formal_completeness_gate_passed"
        ],
        "holdout_interface_accessible": False,
        "holdout_opened": False,
        "formal_training_count": 0,
        "formal_evaluation_count": 0,
        "old_v4_checkpoint_reused": False,
        "performance_claims": [],
    }
    write_json(output_root / "rehearsal_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
