from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    CellTransactionError,
    FormalCellLedger,
    validate_producer_integrity_manifests,
    verify_persisted_committed_phase,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _ledger(root: Path) -> FormalCellLedger:
    return FormalCellLedger(
        run_root=root,
        identity=CellExecutionIdentity(
            run_id=root.name,
            execution_commit="e" * 40,
            protocol_semantic_sha256="p" * 64,
            resource_registry_semantic_sha256="r" * 64,
            environment_fingerprint="v" * 64,
            split_semantic_sha256="s" * 64,
            window_contract_semantic_sha256="w" * 64,
            catalog_fingerprint="c" * 64,
            runtime_identity="t" * 64,
            command_matrix_sha256="m" * 64,
        ),
    )


def _producer(staging: Path, destination: Path) -> Path:
    artifact = staging / "artifact"
    producer = artifact / "benchmark" / "run"
    summary = producer / "episodes" / "one.summary.json"
    _write_json(
        summary,
        {
            "metric": 1.25,
            "nullable_metric": None,
            "row_count": 1,
            "run_info": {"summary_path": str(summary.resolve())},
        },
    )
    _write_json(
        producer / "run_manifest.json",
        {
            "request": {"seed": 7, "agent": "sa_ghmappo"},
            "output_paths": {"episodes": str((producer / "episodes").resolve())},
        },
    )
    (producer / "resolved_command.txt").write_text(
        f"--output_root {producer}\n", encoding="utf-8"
    )
    files = []
    for path in sorted(producer.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(producer).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
    _write_json(
        producer / "artifact_integrity_manifest.json",
        {"integrity_manifest_version": "1.0.0", "files": files},
    )
    return artifact


def test_publication_rebuilds_producer_manifest_before_transaction_inventory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "evaluation_run"
    ledger = _ledger(run_root)
    begun = ledger.begin_cell(
        phase="formal_cache_policy",
        coordinates={"capacity_label": "constrained_288mb"},
        command=["evaluate", "--capacity", "288"],
        input_hash="i" * 64,
        committed_path=run_root / "formal_cache_policy" / "constrained_288mb",
    )
    staging = Path(begun["record"]["staging_path"])
    artifact = _producer(staging, Path(begun["record"]["committed_path"]))
    event = ledger.commit_cell(
        begun["cell_id"],
        validated_artifact_root=artifact,
    )
    destination = Path(event["committed_path"])
    audit = validate_producer_integrity_manifests(destination, require_manifest=True)
    assert audit["status"] == "pass"
    assert audit["manifest_count"] == 1
    marker = json.loads((destination / "committed_marker.json").read_text())
    assert marker["producer_integrity_manifest_count"] == 1
    assert marker["publication_relocation"]["changed_file_count"] == 3
    manifest = json.loads(
        next(destination.glob("benchmark/*/artifact_integrity_manifest.json")).read_text()
    )
    relocation = manifest["publication_relocation"]
    assert relocation["non_path_data_preserved"] is True
    assert all(
        row["non_path_semantic_sha256_before"]
        == row["non_path_semantic_sha256_after"]
        for row in relocation["changed_files"]
    )
    summary = next(destination.glob("benchmark/*/episodes/*.summary.json"))
    payload = json.loads(summary.read_text())
    assert payload["metric"] == 1.25
    assert payload["nullable_metric"] is None
    assert str(destination) in payload["run_info"]["summary_path"]
    ledger.verify_committed(begun["cell_id"])


def test_transaction_inventory_cannot_mask_stale_producer_manifest(tmp_path: Path) -> None:
    root = tmp_path / "published"
    producer = root / "producer"
    _write_json(producer / "value.json", {"metric": 1, "nullable": None})
    _write_json(
        producer / "artifact_integrity_manifest.json",
        {
            "integrity_manifest_version": "1.0.0",
            "files": [
                {
                    "path": "value.json",
                    "size_bytes": (producer / "value.json").stat().st_size,
                    "sha256": _sha(producer / "value.json"),
                }
            ],
        },
    )
    _write_json(producer / "value.json", {"metric": 2, "nullable": None})
    with pytest.raises(CellTransactionError, match="producer integrity payload drift"):
        validate_producer_integrity_manifests(root, require_manifest=True)


def test_controller_layout_allows_transaction_owned_logs_but_still_inventories_them(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "evaluation_run"
    ledger = _ledger(run_root)
    begun = ledger.begin_cell(
        phase="formal_controller",
        coordinates={"capacity_label": "constrained_288mb"},
        command=["evaluate"],
        input_hash="i" * 64,
        committed_path=run_root / "formal_controller" / "constrained_288mb",
    )
    producer = Path(begun["record"]["staging_path"]) / "benchmark_run"
    _write_json(producer / "benchmark_rows.csv.json", {"metric": 1.0})
    payload_path = producer / "benchmark_rows.csv.json"
    _write_json(
        producer / "artifact_integrity_manifest.json",
        {
            "integrity_manifest_version": "1.0.0",
            "files": [{
                "path": payload_path.name,
                "size_bytes": payload_path.stat().st_size,
                "sha256": _sha(payload_path),
            }],
        },
    )
    (producer / "cell_stdout.log").write_text("ok", encoding="utf-8")
    (producer / "cell_stderr.log").write_text("", encoding="utf-8")
    event = ledger.commit_cell(
        begun["cell_id"],
        validated_artifact_root=producer,
        required_paths=[
            "benchmark_rows.csv.json",
            "cell_stdout.log",
            "cell_stderr.log",
        ],
    )
    inventory_paths = {row["path"] for row in event["artifact_inventory"]}
    assert {"cell_stdout.log", "cell_stderr.log"} <= inventory_paths
    ledger.verify_committed(begun["cell_id"])


def test_relocation_preimage_is_revalidated_from_final_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "evaluation_run"
    ledger = _ledger(run_root)
    begun = ledger.begin_cell(
        phase="formal_cache_policy",
        coordinates={"capacity_label": "constrained_288mb"},
        command=["evaluate"],
        input_hash="i" * 64,
        committed_path=run_root / "formal_cache_policy" / "constrained_288mb",
    )
    staging = Path(begun["record"]["staging_path"])
    artifact = _producer(staging, Path(begun["record"]["committed_path"]))
    event = ledger.commit_cell(begun["cell_id"], validated_artifact_root=artifact)
    manifest_path = next(
        Path(event["committed_path"]).glob("benchmark/*/artifact_integrity_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["publication_relocation"]["changed_files"][0]["before_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(CellTransactionError, match="preimage"):
        validate_producer_integrity_manifests(Path(event["committed_path"]))


def test_repeated_statistics_style_consumption_rechecks_both_integrity_layers(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "evaluation_run"
    ledger = _ledger(run_root)
    begun = ledger.begin_cell(
        phase="formal_controller",
        coordinates={"capacity_label": "constrained_288mb"},
        command=["evaluate"],
        input_hash="i" * 64,
        committed_path=run_root / "formal_controller" / "constrained_288mb",
    )
    staging = Path(begun["record"]["staging_path"])
    artifact = _producer(staging, Path(begun["record"]["committed_path"]))
    event = ledger.commit_cell(begun["cell_id"], validated_artifact_root=artifact)
    assert len(
        verify_persisted_committed_phase(run_root, phase="formal_controller")
    ) == 1
    assert len(
        verify_persisted_committed_phase(run_root, phase="formal_controller")
    ) == 1
    summary = next(Path(event["committed_path"]).glob("benchmark/*/episodes/*.summary.json"))
    payload = json.loads(summary.read_text())
    payload["metric"] = 999
    _write_json(summary, payload)
    with pytest.raises(CellTransactionError):
        verify_persisted_committed_phase(run_root, phase="formal_controller")


@pytest.mark.parametrize(
    "mutation",
    [
        {"metric": 1.25, "nullable_metric": 0, "row_count": 1},
        {"metric": 9.0, "nullable_metric": None, "row_count": 1},
        {"metric": 1.25, "nullable_metric": None, "row_count": 2},
    ],
)
def test_non_path_metric_null_and_row_count_changes_are_rejected(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    root = tmp_path / "producer"
    _write_json(root / "value.json", mutation)
    original = {"metric": 1.25, "nullable_metric": None, "row_count": 1}
    expected = tmp_path / "expected.json"
    _write_json(expected, original)
    _write_json(
        root / "artifact_integrity_manifest.json",
        {
            "integrity_manifest_version": "1.0.0",
            "files": [
                {
                    "path": "value.json",
                    "size_bytes": expected.stat().st_size,
                    "sha256": _sha(expected),
                }
            ],
        },
    )
    with pytest.raises(CellTransactionError):
        validate_producer_integrity_manifests(root, require_manifest=True)


@pytest.mark.parametrize("case", ["missing", "duplicate", "escape", "symlink"])
def test_producer_manifest_membership_and_path_boundary_rejections(
    tmp_path: Path, case: str
) -> None:
    root = tmp_path / "producer"
    _write_json(root / "value.json", {"metric": 1})
    row = {
        "path": "value.json",
        "size_bytes": (root / "value.json").stat().st_size,
        "sha256": _sha(root / "value.json"),
    }
    rows = [row]
    if case == "missing":
        rows = []
    elif case == "duplicate":
        rows = [row, row]
    elif case == "escape":
        rows = [{**row, "path": "../value.json"}]
    elif case == "symlink":
        (root / "link.json").symlink_to(root / "value.json")
        rows = [row, {**row, "path": "link.json"}]
    _write_json(
        root / "artifact_integrity_manifest.json",
        {"integrity_manifest_version": "1.0.0", "files": rows},
    )
    with pytest.raises(CellTransactionError):
        validate_producer_integrity_manifests(root, require_manifest=True)
