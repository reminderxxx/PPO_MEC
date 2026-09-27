from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluators.dedicated_holdout_execution import (
    DEDICATED_HOLDOUT_AUTHORIZATION_VERSION,
    DEDICATED_HOLDOUT_CALLER_ROLE,
    DEDICATED_HOLDOUT_ENTRYPOINT,
    DEDICATED_HOLDOUT_EXECUTION_MODE,
    DedicatedHoldoutExecutionError,
    command_semantic_sha256,
    file_sha256,
    load_authorization,
    validate_benchmark_capability,
)
from src.evaluators.formal_window_consumption import (
    FormalWindowConsumptionError,
    build_contract,
    validate_window_plan_binding,
)
from src.evaluators.typed_model_cache_formal_protocol import (
    attach_hashes,
    build_holdout_seal,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _synthetic_bundle(tmp_path: Path) -> dict:
    mobility_path = tmp_path / "synthetic_ngsim.csv"
    rows = ["Vehicle_ID,Frame_ID,Local_X,Local_Y,v_Vel,Location,Global_Time"]
    frame_count = 60 * 24
    for offset in range(frame_count):
        rows.append(
            f"veh-{offset // 80},{offset + 1},{offset % 200},0,10,I-80,{1_000_000_000 + offset * 100}"
        )
    mobility_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    workflow_path = tmp_path / "batch_task.csv"
    workflow_path.write_text(
        "A1,1,j_1,1,Terminated,1,2,100,0.1\n"
        "A2,1,j_1,1,Terminated,2,3,100,0.2\n",
        encoding="utf-8",
    )
    counts = {"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12}
    cursor = 0
    plan_paths: dict[str, Path] = {}
    for split, count in counts.items():
        windows = []
        for local_index in range(count):
            start = cursor * 24
            windows.append(
                {
                    "window_id": f"synthetic-{split}-{local_index:02d}",
                    "frame_offset": start,
                    "window_length": 24,
                    "source_segment_id": "i_80",
                    "source_segment_run_id": "i_80_run_001",
                    "raw_frame_start": start + 1,
                    "raw_frame_end": start + 24,
                    "raw_time_start": 1_000_000_000 + start * 100,
                    "raw_time_end": 1_000_000_000 + (start + 23) * 100,
                    "provider_segment_frame_start": start,
                    "provider_segment_frame_end": start + 23,
                    "sampling_interval": 100,
                    "recommended_rsu_layout": "auto_dominant_tight",
                    "window_class": "synthetic_metadata_only",
                }
            )
            cursor += 1
        plan_path = tmp_path / f"{split}_window_plan.json"
        _write_json(
            plan_path,
            {
                "split": split,
                "protocol_version": "synthetic-interface-test",
                "mobility_source_path": str(mobility_path),
                "outcome_blind_selection": True,
                "selected_window_plan": windows,
            },
        )
        plan_paths[split] = plan_path
    contract = build_contract(
        source_path=mobility_path,
        plan_paths=plan_paths,
        source_row_count=frame_count,
        source_size_bytes=mobility_path.stat().st_size,
        source_sha256=file_sha256(mobility_path),
        provider_frame_count=frame_count,
        split_semantic_sha256="synthetic-split",
        historical_registry_semantic_sha256="synthetic-history",
        inventory_semantic_sha256="synthetic-inventory",
    )
    contract_path = tmp_path / "formal_window_consumption_contract.json"
    _write_json(contract_path, contract)
    split_manifest = attach_hashes(
        {
            "splits": {
                "sealed_holdout": {
                    "outer_window_plan_sha256": file_sha256(
                        plan_paths["sealed_holdout"]
                    ),
                    "outer_window_count": 12,
                }
            }
        }
    )
    return {
        "mobility_path": mobility_path,
        "workflow_path": workflow_path,
        "plan_path": plan_paths["sealed_holdout"],
        "contract": contract,
        "contract_path": contract_path,
        "seal": build_holdout_seal(split_manifest),
    }


def _authorization(
    tmp_path: Path,
    bundle: dict,
    *,
    workflow_path: Path | None = None,
    run_id: str = "synthetic-dedicated-success",
) -> tuple[Path, Path, Path]:
    authorization_path = tmp_path / f"{run_id}.authorization.json"
    opening_path = tmp_path / f"{run_id}.opening.json"
    output_root = tmp_path / "benchmark_output"
    token = "synthetic-one-time-token"
    seal = deepcopy(bundle["seal"])
    seal["one_time_execution_token_status"] = "issued_for_synthetic_test"
    seal["one_time_execution_token_sha256"] = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()
    seal = attach_hashes({key: value for key, value in seal.items() if key != "hashes"})
    argv = [
        "--agents",
        "reactive_lru",
        "--seeds",
        "7",
        "--mobility_source",
        "ngsim",
        "--mobility_csv_path",
        str(bundle["mobility_path"]),
        "--workflow_csv_path",
        str(workflow_path or bundle["workflow_path"]),
        "--max_mobility_rows",
        "1440",
        "--max_workflows",
        "1",
        "--max_steps",
        "1",
        "--min_tasks",
        "1",
        "--max_tasks",
        "10",
        "--workflow_selector",
        "ordered",
        "--rsu_layout",
        "auto_dominant_tight",
        "--primary_vehicle_selection",
        "stable_first",
        "--window_selector",
        "ordered",
        "--window_length",
        "24",
        "--window_plan_path",
        str(bundle["plan_path"]),
        "--formal_window_consumption_contract_path",
        str(bundle["contract_path"]),
        "--formal_window_split",
        "sealed_holdout",
        "--window_consumption_mode",
        DEDICATED_HOLDOUT_EXECUTION_MODE,
        "--holdout-execution-authorization-path",
        str(authorization_path),
        "--holdout-opening-record-path",
        str(opening_path),
        "--output_root",
        str(output_root),
        "--benchmark-run-id",
        run_id,
    ]
    execution_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    authorization = {
        "dedicated_holdout_authorization_version": DEDICATED_HOLDOUT_AUTHORIZATION_VERSION,
        "authorization_id": f"test-{run_id}",
        "state": "issued",
        "execution_authorized": True,
        "caller_role": DEDICATED_HOLDOUT_CALLER_ROLE,
        "target_entrypoint": DEDICATED_HOLDOUT_ENTRYPOINT,
        "allowed_split": "sealed_holdout",
        "window_consumption_mode": DEDICATED_HOLDOUT_EXECUTION_MODE,
        "benchmark_argv": argv,
        "command_semantic_sha256": command_semantic_sha256(argv),
        "opening_gate_results": {
            name: True
            for name in seal["opening_gate"]["allowed_checks"]
        },
        "bindings": {
            "seal_semantic_sha256": seal["hashes"]["semantic_sha256"],
            "window_consumption_contract_semantic_sha256": bundle["contract"][
                "hashes"
            ]["semantic_sha256"],
            "window_plan_file_sha256": file_sha256(bundle["plan_path"]),
            "execution_commit": execution_commit,
            "opening_record_path": str(opening_path),
            "output_run_id": run_id,
            "output_root": str(output_root),
            "candidate_checkpoint_manifest_sha256": hashlib.sha256(
                b"checkpoint-free-synthetic-test"
            ).hexdigest(),
            "statistics_contract_sha256": hashlib.sha256(
                b"synthetic-statistics-contract"
            ).hexdigest(),
        },
    }
    authorization = attach_hashes(authorization)
    _write_json(authorization_path, authorization)
    seal_path = tmp_path / f"{run_id}.seal.json"
    token_path = tmp_path / f"{run_id}.token"
    _write_json(seal_path, seal)
    token_path.write_text(token, encoding="utf-8")
    return authorization_path, seal_path, token_path


def test_ordinary_and_identity_only_paths_cannot_execute_sealed_holdout(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    common = dict(
        contract=bundle["contract"],
        plan_path=bundle["plan_path"],
        split="sealed_holdout",
        max_mobility_rows=1440,
        mobility_csv_path=bundle["mobility_path"],
        window_selector="ordered",
        window_length=24,
        rsu_layout="auto_dominant_tight",
        primary_vehicle_selection="stable_first",
    )
    identity = validate_window_plan_binding(**common, mode="identity_only")
    assert identity["status"] == "pass"
    with pytest.raises(FormalWindowConsumptionError, match="authorized dedicated"):
        validate_window_plan_binding(**common, mode="formal")
    with pytest.raises(FormalWindowConsumptionError, match="capability"):
        validate_window_plan_binding(
            **common, mode=DEDICATED_HOLDOUT_EXECUTION_MODE
        )


def test_actual_benchmark_preflight_rejects_unopened_authorization(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    authorization_path, _, _ = _authorization(
        tmp_path, bundle, run_id="synthetic-unopened-rejection"
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark_main_results.py"),
            *authorization["benchmark_argv"],
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert "unable to load holdout opening record" in completed.stderr
    assert not Path(authorization["bindings"]["output_root"]).exists()


def test_checkpoint_and_statistics_bindings_require_real_sha256(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    authorization_path, _, _ = _authorization(
        tmp_path, bundle, run_id="synthetic-invalid-binding-hash"
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["bindings"]["statistics_contract_sha256"] = "not-a-hash"
    authorization = attach_hashes(
        {key: value for key, value in authorization.items() if key != "hashes"}
    )
    _write_json(authorization_path, authorization)
    with pytest.raises(DedicatedHoldoutExecutionError, match="64-character"):
        load_authorization(authorization_path)


def test_authorized_production_command_runs_actual_scientific_entrypoint(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    authorization_path, seal_path, token_path = _authorization(tmp_path, bundle)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_dedicated_holdout_benchmark.py"),
            "--authorization-path",
            str(authorization_path),
            "--seal-path",
            str(seal_path),
            "--execution-token-file",
            str(token_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    opening_path = Path(authorization["bindings"]["opening_record_path"])
    opening = json.loads(opening_path.read_text(encoding="utf-8"))
    assert opening["consumed_permanently"] is True
    aggregate = (
        Path(authorization["bindings"]["output_root"])
        / authorization["bindings"]["output_run_id"]
        / "aggregate_summary.json"
    )
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert payload["formal_window_consumption_binding"]["mode"] == (
        DEDICATED_HOLDOUT_EXECUTION_MODE
    )
    assert payload["dedicated_holdout_capability_validation"]["status"] == "pass"
    assert len(payload["selected_window_plan"]) == 12
    assert payload["agents"] == ["reactive_lru"]


def test_identity_conflict_is_rejected_before_scientific_execution(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    authorization_path, _, _ = _authorization(
        tmp_path, bundle, run_id="synthetic-identity-conflict"
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    opening_path = Path(authorization["bindings"]["opening_record_path"])
    opening = {
        "consumed_permanently": True,
        "authorization_semantic_sha256": authorization["hashes"]["semantic_sha256"],
        "command_semantic_sha256": authorization["command_semantic_sha256"],
        "output_run_id": authorization["bindings"]["output_run_id"],
        "command": authorization["benchmark_argv"],
    }
    _write_json(opening_path, opening)
    plan = json.loads(bundle["plan_path"].read_text(encoding="utf-8"))
    plan["selected_window_plan"][0]["raw_time_start"] += 100
    _write_json(bundle["plan_path"], plan)
    with pytest.raises(DedicatedHoldoutExecutionError, match="plan identity mismatch"):
        validate_benchmark_capability(
            authorization_path=authorization_path,
            opening_record_path=opening_path,
            actual_benchmark_argv=authorization["benchmark_argv"],
            contract_semantic_sha256=bundle["contract"]["hashes"][
                "semantic_sha256"
            ],
            plan_path=bundle["plan_path"],
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark_main_results.py"),
            *authorization["benchmark_argv"],
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert "holdout window plan identity mismatch" in completed.stderr
    assert not Path(authorization["bindings"]["output_root"]).exists()


def test_post_open_failure_is_permanently_recorded_and_cannot_retry(
    tmp_path: Path,
) -> None:
    bundle = _synthetic_bundle(tmp_path)
    missing_workflow = tmp_path / "missing.csv"
    authorization_path, seal_path, token_path = _authorization(
        tmp_path,
        bundle,
        workflow_path=missing_workflow,
        run_id="synthetic-post-open-failure",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/run_dedicated_holdout_benchmark.py"),
        "--authorization-path",
        str(authorization_path),
        "--seal-path",
        str(seal_path),
        "--execution-token-file",
        str(token_path),
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert first.returncode != 0
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    opening_path = Path(authorization["bindings"]["opening_record_path"])
    assert json.loads(opening_path.read_text(encoding="utf-8"))[
        "consumed_permanently"
    ] is True
    second = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert second.returncode != 0
    assert "already exists" in second.stderr
