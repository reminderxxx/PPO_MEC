from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_typed_model_cache_formal_protocol import (
    validate_complete_without_holdout_gate,
)
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    CellTransactionError,
    FormalCellLedger,
    resolve_child_output_descriptor,
    single_child_directory,
    write_child_output_descriptor,
)
from src.runtime.formal_protocol_capabilities import (
    ACTIVE_EXECUTION_PROTOCOL_VERSION,
    get_protocol_capabilities,
)


ROOT = Path(__file__).resolve().parents[1]
V26 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_6_20260905/protocol_v2_6_manifest.json"


def identity(run_id: str = "run") -> CellExecutionIdentity:
    return CellExecutionIdentity(
        run_id=run_id,
        execution_commit="a" * 40,
        protocol_semantic_sha256="b" * 64,
        resource_registry_semantic_sha256="c" * 64,
        environment_fingerprint="d" * 64,
        split_semantic_sha256="e" * 64,
        window_contract_semantic_sha256="f" * 64,
        catalog_fingerprint="1" * 64,
        runtime_identity="2" * 64,
        command_matrix_sha256="3" * 64,
    )


def prepare_support_payload(
    ledger: FormalCellLedger, *, setting: str = "setting-a"
) -> tuple[dict, Path, Path]:
    coordinates = {"support_setting_id": setting}
    begun = ledger.begin_cell(
        phase="formal_support",
        coordinates=coordinates,
        command=["support", setting],
        input_hash="input",
        committed_path=ledger.run_root / "formal_support" / setting,
    )
    staging = Path(begun["record"]["staging_path"])
    child = staging / "child_output"
    payload = child / "main_results_exact"
    payload.mkdir(parents=True)
    (payload / "support_provenance.json").write_text(
        json.dumps({"setting_id": setting, "output": str(payload)}), encoding="utf-8"
    )
    (payload / "aggregate_summary.json").write_text("{}", encoding="utf-8")
    (payload / "benchmark_rows.csv").write_text("x\n1\n", encoding="utf-8")
    descriptor = child / "cell_child_output.json"
    write_child_output_descriptor(
        descriptor,
        cell_id=begun["cell_id"],
        phase="formal_support",
        logical_setting_id=setting,
        output_root=child,
        artifact_root=payload,
        producer_kind="benchmark_support",
        required_payload=[
            "support_provenance.json", "aggregate_summary.json", "benchmark_rows.csv"
        ],
    )
    return begun, child, descriptor


def test_protocol_v26_is_unique_active_and_v25_is_historical() -> None:
    protocol = json.loads(V26.read_text(encoding="utf-8-sig"))
    assert ACTIVE_EXECUTION_PROTOCOL_VERSION == "2.6.0"
    assert protocol["typed_model_cache_formal_protocol_version"] == "2.6.0"
    assert get_protocol_capabilities("2.6.0").cell_artifact_publication_required
    assert get_protocol_capabilities("2.6.0").live_execution_allowed
    assert not get_protocol_capabilities("2.5.0").live_execution_allowed


def test_structured_support_output_is_atomically_published_and_rebased(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun, child, descriptor = prepare_support_payload(ledger)
    payload, contract = resolve_child_output_descriptor(
        descriptor,
        output_root=child,
        expected_cell_id=begun["cell_id"],
        expected_phase="formal_support",
        expected_setting_id="setting-a",
    )
    (payload / "cell_stdout.log").write_text("ok", encoding="utf-8")
    (payload / "cell_stderr.log").write_text("", encoding="utf-8")
    committed = ledger.commit_cell(
        begun["cell_id"],
        required_paths=[*contract["required_payload"], "cell_stdout.log", "cell_stderr.log"],
        validated_artifact_root=payload,
        child_output_path=child,
    )
    root = Path(committed["committed_path"])
    provenance = json.loads((root / "support_provenance.json").read_text())
    assert provenance["output"] == str(root)
    assert (root / "committed_marker.json").is_file()
    assert ledger.verify_committed(begun["cell_id"])["status"] == "committed"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_setting", "identity mismatch"),
        ("cross_cell", "identity mismatch"),
        ("inventory_drift", "inventory drift"),
        ("conflicting_output", "conflicting outputs"),
    ],
)
def test_child_output_identity_conflict_and_drift_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun, child, descriptor = prepare_support_payload(ledger)
    expected_cell = begun["cell_id"]
    expected_setting = "setting-a"
    if mutation == "wrong_setting":
        expected_setting = "setting-b"
    elif mutation == "cross_cell":
        expected_cell = "other-cell"
    elif mutation == "inventory_drift":
        (child / "main_results_exact" / "extra.json").write_text("{}")
    else:
        (child / "other-output").mkdir()
    with pytest.raises(CellTransactionError, match=message):
        resolve_child_output_descriptor(
            descriptor,
            output_root=child,
            expected_cell_id=expected_cell,
            expected_phase="formal_support",
            expected_setting_id=expected_setting,
        )


def test_missing_payload_symlink_escape_and_duplicate_directory_rejected(tmp_path: Path) -> None:
    only = tmp_path / "one"
    only.mkdir()
    (only / "a").mkdir()
    (only / "b").mkdir()
    with pytest.raises(CellTransactionError, match="exactly one"):
        single_child_directory(only)
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun, child, descriptor = prepare_support_payload(ledger)
    payload = child / "main_results_exact"
    (payload / "benchmark_rows.csv").unlink()
    with pytest.raises(CellTransactionError, match="inventory drift|missing"):
        resolve_child_output_descriptor(
            descriptor,
            output_root=child,
            expected_cell_id=begun["cell_id"],
            expected_phase="formal_support",
            expected_setting_id="setting-a",
        )
    link = payload / "escape"
    link.symlink_to(tmp_path)
    with pytest.raises(CellTransactionError, match="symlink"):
        write_child_output_descriptor(
            child / "other.json",
            cell_id=begun["cell_id"],
            phase="formal_support",
            logical_setting_id="setting-a",
            output_root=child,
            artifact_root=payload,
            producer_kind="benchmark_support",
            required_payload=["support_provenance.json"],
        )


def test_gate_false_process_exits_nonzero_even_after_json_is_written(tmp_path: Path) -> None:
    output = tmp_path / "formal_gate.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py"),
        "--action", "formal_gate",
        "--protocol-path", str(
            ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/protocol_v1_1_manifest.json"
        ),
        "--input-root", str(tmp_path),
        "--output-path", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 2
    assert json.loads(output.read_text())["passed"] is False


def test_complete_without_holdout_requires_passed_gate_and_completed_terminal(tmp_path: Path) -> None:
    protocol = {"hashes": {"semantic_sha256": "a" * 64}}
    (tmp_path / "formal_gate.json").write_text(
        json.dumps({
            "passed": False,
            "protocol_semantic_sha256": "a" * 64,
            "completeness_only": True,
            "performance_threshold_used": False,
            "holdout_opened": False,
            "exact_count_status": "pass",
            "missing_outputs": [],
            "exact_count_mismatches": {},
            "generated_checkpoint_registry_audit": {"status": "pass"},
        })
    )
    (tmp_path / "phase_state.jsonl").write_text("")
    with pytest.raises(ValueError, match="failed formal gate"):
        validate_complete_without_holdout_gate(tmp_path, protocol)
