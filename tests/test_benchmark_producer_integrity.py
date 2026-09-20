from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from src.evaluators.cache_baseline_fairness import build_manifest
from scripts.run_typed_model_cache_formal_support import (
    refresh_oracle_producer_manifest,
    stamp_outputs,
)
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    CellTransactionError,
    FormalCellLedger,
    execute_cell_artifact_transaction,
    resolve_child_output_descriptor,
    validate_producer_integrity_manifest,
    validate_producer_integrity_manifests,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("PPO_MEC_FROZEN_PYTHON", sys.executable)).absolute()
DRIVER = ROOT / "tests/benchmark_producer_publication_driver.py"
DIAGNOSTICS = {
    "comparison_against_popularity.json",
    "sa_advantage_diagnosis.json",
}
CORE_PAYLOAD = {
    "aggregate_summary.json",
    "benchmark_rows.csv",
    "comparison_against_popularity.json",
    "sa_advantage_diagnosis.json",
    "run_manifest.json",
    "resolved_command.txt",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_stamp_rebuilds_final_nested_producer_manifest(tmp_path: Path) -> None:
    run = tmp_path / "benchmark"
    episode = run / "episodes/window/workflow/ppo/seed_7.summary.json"
    episode.parent.mkdir(parents=True)
    _write_json(episode, {"run_info": {"summary_path": str(episode)}})
    _write_json(run / "aggregate_summary.json", {"episode_count": 1})
    (run / "benchmark_rows.csv").write_text("agent_name,seed\nppo,7\n")
    _write_json(run / "artifact_integrity_manifest.json", {
        "integrity_manifest_version": "1.0.0",
        "files": [{"path": path.relative_to(run).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha(path)}
                  for path in sorted(run.rglob("*")) if path.is_file()],
    })
    stamp_outputs(run, {"support_family": "ablation", "setting_id": "nonformal", "support_setting_sha256": "s" * 64, "protocol_semantic_sha256": "p" * 64, "split_semantic_sha256": "w" * 64})
    audit = validate_producer_integrity_manifest(run / "artifact_integrity_manifest.json")
    assert audit["file_count"] == 4


def test_oracle_support_provenance_is_in_final_producer_manifest(tmp_path: Path) -> None:
    run = tmp_path / "oracle"
    run.mkdir()
    _write_json(run / "oracle_results.json", {"horizons": [1, 3]})
    _write_json(run / "artifact_integrity_manifest.json", {
        "artifact_integrity_manifest_version": "1.0.0",
        "manifest_validation_status": "pass",
        "request_replay_validation_status": "pass",
        "files": [{"path": "oracle_results.json", "size_bytes": (run / "oracle_results.json").stat().st_size,
                   "sha256": _sha(run / "oracle_results.json")}],
    })
    _write_json(run / "support_provenance.json", {"setting_id": "scalability-b05d0a8a8684c1f0"})
    with pytest.raises(CellTransactionError, match="membership drift"):
        validate_producer_integrity_manifest(run / "artifact_integrity_manifest.json")
    refresh_oracle_producer_manifest(run)
    audit = validate_producer_integrity_manifest(run / "artifact_integrity_manifest.json")
    assert {row["path"] for row in audit["files"]} == {
        "oracle_results.json", "support_provenance.json"
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _identity(run_id: str) -> CellExecutionIdentity:
    return CellExecutionIdentity(
        run_id=run_id,
        execution_commit="e" * 40,
        protocol_semantic_sha256="p" * 64,
        resource_registry_semantic_sha256="r" * 64,
        environment_fingerprint="v" * 64,
        split_semantic_sha256="s" * 64,
        window_contract_semantic_sha256="w" * 64,
        catalog_fingerprint="c" * 64,
        runtime_identity="t" * 64,
        command_matrix_sha256="m" * 64,
    )


@pytest.fixture(scope="module")
def controlled_inputs() -> dict[str, Path]:
    (ROOT / "artifacts").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="g14r20_i3_real_benchmark_", dir=ROOT / "artifacts"))
    mobility = root / "ngsim.csv"
    with mobility.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel", "Location", "Global_Time"],
        )
        writer.writeheader()
        for frame in range(24):
            for vehicle in range(2):
                writer.writerow({
                    "Vehicle_ID": vehicle + 1,
                    "Frame_ID": frame,
                    "Local_X": vehicle * 4,
                    "Local_Y": frame + vehicle,
                    "v_Vel": 10 + vehicle,
                    "Location": "controlled",
                    "Global_Time": 1000 + frame,
                })
    workflow = root / "batch_task.csv"
    workflow.write_text(
        "M1,1,j_1,1,Terminated,0,1,1,0.1\n"
        "R2_1,1,j_1,1,Terminated,1,2,1,0.1\n"
        "R3_2,1,j_1,1,Terminated,2,3,1,0.1\n"
        "R4_3,1,j_1,1,Terminated,3,4,1,0.1\n"
        "R5_4,1,j_1,1,Terminated,4,5,1,0.1\n",
        encoding="utf-8",
    )
    plan = root / "window_plan.json"
    _write_json(plan, {
        "protocol_version": "g14r20_i3_nonformal_fixture_v1",
        "split": "controlled_nonformal",
        "sealed": False,
        "outcome_blind_selection": True,
        "mobility_source_path": str(mobility),
        "selected_window_plan": [{
            "window_rank": 0,
            "window_id": "window_controlled_off0_len24_t1000_1023",
            "frame_offset": 0,
            "window_length": 24,
            "time_index_start": 1000,
            "time_index_end": 1023,
            "source_segment_id": "controlled",
            "source_location": "controlled",
            "segment_frame_start": 0,
            "segment_frame_end": 23,
            "dominant_axis": "y",
            "recommended_rsu_layout": "auto_dominant_tight",
            "chosen_rsu_axis": "y",
            "coverage_radius": 8.0,
            "spacing": 9.0,
            "estimated_association_change_count": 1,
            "estimated_handoff_count": 1,
            "window_class": "controlled_nonformal",
        }],
    })
    fairness = root / "fairness.json"
    _write_json(fairness, build_manifest(
        root=ROOT,
        mobility_path=mobility,
        workflow_path=workflow,
        window_plan_path=plan,
        catalog_path=ROOT / "src/data/model_catalog/sample_model_catalog.json",
        seeds=[7],
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=48,
        primary_vehicle_selection="stable_first",
        capacity_unit="adapter_slots",
        capacity_value=3,
        output_root=str(root / "fairness_output"),
        evaluation_unit_limit=1,
        created_at="2026-09-13T00:00:00Z",
    ))
    yield {"root": root, "mobility": mobility, "workflow": workflow, "plan": plan, "fairness": fairness}
    shutil.rmtree(root)


def _benchmark_args(inputs: dict[str, Path], output: Path, *, fairness: bool) -> list[str]:
    args = [
        "--agents", "reactive_lru",
        "--seeds", "7",
        "--mobility_csv_path", str(inputs["mobility"]),
        "--workflow_csv_path", str(inputs["workflow"]),
        "--window_plan_path", str(inputs["plan"]),
        "--max_mobility_rows", "48",
        "--max_workflows", "1",
        "--max_steps", "1",
        "--min_tasks", "5",
        "--max_tasks", "20",
        "--workflow_selector", "ordered",
        "--primary_vehicle_selection", "stable_first",
        "--non-formal-rehearsal",
        "--output_root", str(output),
    ]
    if fairness:
        args.extend(["--cache_baseline_fairness_manifest_path", str(inputs["fairness"])])
    return args


def _run_producer(inputs: dict[str, Path], output: Path, *, fairness: bool) -> Path:
    completed = subprocess.run(
        [str(PYTHON), "-B", str(ROOT / "scripts/benchmark_main_results.py"), *_benchmark_args(inputs, output, fairness=fairness)],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    runs = [path for path in output.iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _assert_exact_manifest(producer: Path, *, fairness: bool) -> dict[str, object]:
    manifest_path = producer / "artifact_integrity_manifest.json"
    audit = validate_producer_integrity_manifest(manifest_path)
    declared = {row["path"]: row for row in audit["files"]}
    observed = {
        path.relative_to(producer).as_posix(): path
        for path in producer.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert set(declared) == set(observed)
    assert CORE_PAYLOAD <= set(declared)
    assert DIAGNOSTICS <= set(declared)
    assert any(name.startswith("episodes/") for name in declared)
    if fairness:
        assert {"cache_baseline_fairness_manifest.json", "fairness_runtime_audit.json"} <= set(declared)
    else:
        assert "cache_baseline_fairness_manifest.json" not in declared
        assert "fairness_runtime_audit.json" not in declared
    for name, path in observed.items():
        assert declared[name] == {"path": name, "size_bytes": path.stat().st_size, "sha256": _sha(path)}
    return audit


@pytest.mark.parametrize("fairness", [False, True], ids=["without_fairness", "with_fairness"])
def test_real_benchmark_main_manifest_is_exact(
    controlled_inputs: dict[str, Path], tmp_path: Path, fairness: bool
) -> None:
    producer = _run_producer(controlled_inputs, tmp_path / "benchmark", fairness=fairness)
    _assert_exact_manifest(producer, fairness=fairness)


def test_real_descriptor_transaction_publication_and_readback(
    controlled_inputs: dict[str, Path], tmp_path: Path
) -> None:
    run_root = tmp_path / "transaction_run"
    ledger = FormalCellLedger(run_root=run_root, identity=_identity(run_root.name))
    phase = "formal_cache_policy"
    setting = "controlled_nonformal"

    def build(staging: Path, cell_id: str) -> list[str]:
        return [
            str(PYTHON), "-B", str(DRIVER),
            "--staging", str(staging),
            "--cell-id", cell_id,
            "--phase", phase,
            "--setting-id", setting,
            *_benchmark_args(controlled_inputs, staging / "artifact/benchmark", fairness=True),
        ]

    prepublication: dict[str, str] = {}

    def resolve(staging: Path, cell_id: str, _completed: subprocess.CompletedProcess[str]):
        artifact, descriptor = resolve_child_output_descriptor(
            staging / "cell_child_output.json",
            output_root=staging,
            expected_cell_id=cell_id,
            expected_phase=phase,
            expected_setting_id=setting,
        )
        producer = next((artifact / "benchmark").iterdir())
        _assert_exact_manifest(producer, fairness=True)
        prepublication["rows"] = _sha(producer / "benchmark_rows.csv")
        prepublication["aggregate"] = _sha(producer / "aggregate_summary.json")
        return artifact, descriptor["required_payload"], staging / "cell_child_output.json"

    result = execute_cell_artifact_transaction(
        ledger,
        phase=phase,
        coordinates={"capacity_label": setting},
        command=["real-benchmark", setting],
        input_hash="i" * 64,
        committed_path=run_root / phase / setting,
        command_builder=build,
        artifact_resolver=resolve,
        environment=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"),
        cwd=ROOT,
    )
    assert result["status"] == "committed"
    destination = Path(result["record"]["committed_path"])
    producer = next((destination / "benchmark").iterdir())
    assert _sha(producer / "benchmark_rows.csv") == prepublication["rows"]
    assert _sha(producer / "aggregate_summary.json") == prepublication["aggregate"]
    assert validate_producer_integrity_manifests(destination, require_manifest=True)["status"] == "pass"
    first_inventory = {path.relative_to(destination).as_posix(): _sha(path) for path in destination.rglob("*") if path.is_file()}
    ledger.verify_committed(result["cell_id"])
    assert validate_producer_integrity_manifests(destination, require_manifest=True)["status"] == "pass"
    ledger.verify_committed(result["cell_id"])
    assert first_inventory == {path.relative_to(destination).as_posix(): _sha(path) for path in destination.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "case",
    ["missing_diagnostic_member", "missing_file", "content_drift", "extra_file", "duplicate_member"],
)
def test_real_producer_negative_cases_fail_terminal_without_commit(
    controlled_inputs: dict[str, Path], tmp_path: Path, case: str
) -> None:
    run_root = tmp_path / case
    ledger = FormalCellLedger(run_root=run_root, identity=_identity(run_root.name))
    phase = "formal_cache_policy"

    def build(staging: Path, cell_id: str) -> list[str]:
        return [
            str(PYTHON), "-B", str(DRIVER),
            "--staging", str(staging),
            "--cell-id", cell_id,
            "--phase", phase,
            "--setting-id", case,
            *_benchmark_args(controlled_inputs, staging / "artifact/benchmark", fairness=False),
        ]

    def resolve(staging: Path, cell_id: str, _completed: subprocess.CompletedProcess[str]):
        artifact, descriptor = resolve_child_output_descriptor(
            staging / "cell_child_output.json",
            output_root=staging,
            expected_cell_id=cell_id,
            expected_phase=phase,
            expected_setting_id=case,
        )
        producer = next((artifact / "benchmark").iterdir())
        manifest_path = producer / "artifact_integrity_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if case == "missing_diagnostic_member":
            manifest["files"] = [row for row in manifest["files"] if row["path"] != "sa_advantage_diagnosis.json"]
            _write_json(manifest_path, manifest)
        elif case == "missing_file":
            (producer / "sa_advantage_diagnosis.json").unlink()
        elif case == "content_drift":
            (producer / "sa_advantage_diagnosis.json").write_text("{}\n", encoding="utf-8")
        elif case == "extra_file":
            (producer / "unregistered.json").write_text("{}\n", encoding="utf-8")
        elif case == "duplicate_member":
            manifest["files"].append(dict(manifest["files"][0]))
            _write_json(manifest_path, manifest)
        return artifact, descriptor["required_payload"], staging / "cell_child_output.json"

    destination = run_root / phase / case
    with pytest.raises(CellTransactionError):
        execute_cell_artifact_transaction(
            ledger,
            phase=phase,
            coordinates={"capacity_label": case},
            command=["real-benchmark", case],
            input_hash="i" * 64,
            committed_path=destination,
            command_builder=build,
            artifact_resolver=resolve,
            environment=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"),
            cwd=ROOT,
        )
    records = ledger.records()
    assert records[-1]["status"] == "failed_terminal"
    assert records[-1]["failure_classification"] == "cell_artifact_publication_validation_failure"
    assert not destination.exists()
    assert not list(run_root.rglob("committed_marker.json"))
