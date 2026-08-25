from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.run_typed_model_cache_formal_protocol as protocol_runner
import scripts.validate_typed_model_cache_formal_restart as restart_validator
from scripts.run_typed_model_cache_formal_protocol import (
    reject_invalid_run_root,
    resolved_expansion_context,
)
from scripts.manage_typed_model_cache_formal_artifacts import checkpoint_freeze
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    READY_V7_VERDICT,
    readiness_v7,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
from src.runtime.resolved_formal_execution_context import (
    ResolvedFormalExecutionContextError,
    atomic_create_resolved_formal_execution_context,
    build_resolved_formal_execution_context,
    canonical_sha256,
    load_resolved_formal_execution_context,
    resolved_python_for_nested_consumer,
    validate_resolved_formal_execution_context,
)


ROOT = Path(__file__).resolve().parents[1]
V14 = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_4_20260825"
    / "protocol_v1_4_manifest.json"
)
V15_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_5_20260825"
V15 = V15_ROOT / "protocol_v1_5_manifest.json"
ENVIRONMENT = V15_ROOT / "execution_environment_manifest.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture()
def protocol() -> dict:
    payload = load(V15)
    assert validate_protocol_v1_1(payload)["status"] == "pass"
    return payload


@pytest.fixture()
def built_context(protocol: dict, tmp_path: Path) -> tuple[dict, dict, dict]:
    output_root = tmp_path / "durable-run"
    expansion = resolved_expansion_context(
        protocol,
        protocol_path=str(V15),
        output_root=str(output_root),
        python_executable=sys.executable,
    )
    expansion_report = validate_command_templates(
        protocol["execution_contract"]["command_templates"], expansion
    )
    environment_identity = protocol["formal_execution_environment_contract"][
        "scientific_identity"
    ]
    runtime_audit = {
        "observed_execution_commit": "a" * 40,
        "resolved_python_absolute_path": str(Path(sys.executable).absolute()),
        "resolution_source": "explicit_python_executable",
    }
    payload = build_resolved_formal_execution_context(
        protocol=protocol,
        expansion_context=expansion,
        environment_identity=environment_identity,
        runtime_audit=runtime_audit,
        environment_manifest_path=ENVIRONMENT,
        outer_expansion_sha256=expansion_report["command_matrix_sha256"],
        phase_count=expansion_report["phase_count"],
        command_count=expansion_report["command_count"],
    )
    return payload, expansion_report, runtime_audit


def rehash(payload: dict) -> dict:
    result = deepcopy(payload)
    result["context_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "context_sha256"}
    )
    return result


def test_v14_exact_unresolved_python_failure_is_reproduced() -> None:
    old = load(V14)
    with pytest.raises(
        FormalExecutionError, match="unresolved command placeholder: python_executable"
    ):
        validate_command_templates(
            old["execution_contract"]["command_templates"],
            old["execution_contract"]["default_expansion_context"],
        )


def test_v15_outer_and_nested_expansion_are_byte_identical(
    protocol: dict, built_context: tuple[dict, dict, dict]
) -> None:
    payload, outer, _ = built_context
    nested = validate_command_templates(
        protocol["execution_contract"]["command_templates"],
        payload["resolved_expansion_context"],
    )
    assert canonical_sha256(nested["canonical_expansion"]) == canonical_sha256(
        outer["canonical_expansion"]
    )
    assert nested["command_matrix_sha256"] == outer["command_matrix_sha256"]
    assert nested["phase_count"] == 15
    assert nested["command_count"] == 186
    assert all(
        "/ABSOLUTE/" not in token
        for phase in nested["expanded"].values()
        for row in phase["commands"]
        for token in row
    )
    assert all(
        row[0] == str(Path(sys.executable).absolute())
        for phase in nested["expanded"].values()
        for row in phase["commands"]
    )


def test_context_create_only_load_and_missing_rejection(
    protocol: dict, built_context: tuple[dict, dict, dict], tmp_path: Path
) -> None:
    payload, _, _ = built_context
    target = Path(payload["runtime_location"]["resolved_execution_context_path"])
    with pytest.raises(ResolvedFormalExecutionContextError, match="required"):
        load_resolved_formal_execution_context(tmp_path / "missing.json")
    report = atomic_create_resolved_formal_execution_context(target, payload)
    observed, loaded = load_resolved_formal_execution_context(
        target,
        protocol=protocol,
        clean_worktree_root=ROOT,
        durable_run_root=target.parent,
    )
    assert observed == payload
    assert loaded["file_sha256"] == report["file_sha256"]
    with pytest.raises(ResolvedFormalExecutionContextError, match="already exists"):
        atomic_create_resolved_formal_execution_context(target, payload)


def test_v15_active_preflight_rejects_missing_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_typed_model_cache_formal_restart.py",
            "--protocol-path",
            str(V15),
            "--output-path",
            str(tmp_path / "preflight.json"),
            "--window-consumption-contract-path",
            str(
                ROOT
                / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
                / "formal_window_consumption_contract.json"
            ),
        ],
    )
    with pytest.raises(ValueError, match="requires resolved execution context"):
        restart_validator.main()


def test_v15_outer_runner_rejects_implicit_python_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_typed_model_cache_formal_protocol.py",
            "--protocol-path",
            str(V15),
            "--output-root",
            str(tmp_path / "run"),
            "--preflight",
            "--dry-run",
        ],
    )
    with pytest.raises(FormalExecutionError, match="requires explicit Python"):
        protocol_runner.main()


def test_context_sha_tamper_rejected(
    built_context: tuple[dict, dict, dict]
) -> None:
    payload, _, _ = built_context
    payload["context_sha256"] = "0" * 64
    with pytest.raises(ResolvedFormalExecutionContextError, match="SHA-256"):
        validate_resolved_formal_execution_context(payload)


def test_python_path_and_implicit_nested_fallback_rejected(
    built_context: tuple[dict, dict, dict]
) -> None:
    payload, _, runtime = built_context
    drift = dict(runtime, resolved_python_absolute_path="/usr/bin/python3")
    with pytest.raises(ResolvedFormalExecutionContextError, match="Python path drift"):
        validate_resolved_formal_execution_context(payload, runtime_audit=drift)
    with pytest.raises(ResolvedFormalExecutionContextError, match="nested consumer Python"):
        resolved_python_for_nested_consumer(
            payload, observed_sys_executable="/usr/bin/python3"
        )


def test_environment_and_dependency_fingerprint_drift_rejected(
    built_context: tuple[dict, dict, dict]
) -> None:
    payload, _, _ = built_context
    scientific = payload["scientific_identity"]
    for field in ("environment_fingerprint", "dependency_fingerprint"):
        identity = {
            "environment_fingerprint": scientific["environment_fingerprint"],
            "dependency_fingerprint": scientific["dependency_fingerprint"],
        }
        identity[field] = "f" * 64
        with pytest.raises(ResolvedFormalExecutionContextError, match=field):
            validate_resolved_formal_execution_context(
                payload, environment_identity=identity
            )


def test_execution_commit_drift_rejected(
    built_context: tuple[dict, dict, dict]
) -> None:
    payload, _, runtime = built_context
    with pytest.raises(ResolvedFormalExecutionContextError, match="commit drift"):
        validate_resolved_formal_execution_context(
            payload, runtime_audit=dict(runtime, observed_execution_commit="b" * 40)
        )


def test_protocol_semantic_drift_rejected(
    protocol: dict, built_context: tuple[dict, dict, dict]
) -> None:
    payload, _, _ = built_context
    mutated = deepcopy(protocol)
    mutated["status"] = "semantically-mutated"
    mutated = attach_hashes(mutated)
    with pytest.raises(ResolvedFormalExecutionContextError, match="protocol identity drift"):
        validate_resolved_formal_execution_context(payload, protocol=mutated)


def test_cross_run_and_clean_root_drift_rejected(
    built_context: tuple[dict, dict, dict], tmp_path: Path
) -> None:
    payload, _, _ = built_context
    with pytest.raises(ResolvedFormalExecutionContextError, match="run root drift"):
        validate_resolved_formal_execution_context(
            payload, durable_run_root=tmp_path / "another-run"
        )
    with pytest.raises(ResolvedFormalExecutionContextError, match="worktree root drift"):
        validate_resolved_formal_execution_context(
            payload, clean_worktree_root=tmp_path
        )


def test_absolute_sentinel_and_relative_venv_rejected(
    built_context: tuple[dict, dict, dict]
) -> None:
    payload, _, _ = built_context
    sentinel = deepcopy(payload)
    sentinel["resolved_expansion_context"]["output_root"] = (
        "/ABSOLUTE/FORMAL_OUTPUT_ROOT"
    )
    sentinel = rehash(sentinel)
    with pytest.raises(ResolvedFormalExecutionContextError, match="sentinel"):
        validate_resolved_formal_execution_context(sentinel)
    relative = deepcopy(payload)
    relative["runtime_location"]["resolved_python_absolute_path"] = (
        ".venv/bin/python"
    )
    relative = rehash(relative)
    with pytest.raises(ResolvedFormalExecutionContextError, match="not absolute"):
        validate_resolved_formal_execution_context(relative)


def test_v15_invalid_runs_and_holdout_boundary(protocol: dict) -> None:
    failures = protocol["supersession"]["invalid_execution_runs"]
    assert len(failures) == 6
    assert failures[-1]["run_id"].endswith("g14c_v5")
    assert all(not row["resume_allowed"] for row in failures)
    assert all(not row["checkpoint_reuse_allowed"] for row in failures)
    assert all(not row["legacy_phase_finalize_allowed"] for row in failures)
    assert protocol["holdout_execution_contract"]["sealed"] is True
    assert protocol["holdout_execution_contract"]["opened"] is False


def test_v5_resume_finalize_and_checkpoint_references_are_rejected(
    protocol: dict, tmp_path: Path
) -> None:
    invalid_root = (
        tmp_path
        / "typed_model_cache_formal_20260825_111625_g14c_v5"
    )
    for _mode in ("fresh", "resume", "finalize-only"):
        with pytest.raises(FormalExecutionError, match="permanently rejected"):
            reject_invalid_run_root(protocol, invalid_root)
    selection = {
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selected": [
            {
                "agent_name": "ppo",
                "seed": 7,
                "capacity_label": "medium_576mb",
                "update_index": 4,
                "checkpoint_path": str(
                    ROOT
                    / "artifacts/experiments/typed_model_cache_formal"
                    / "typed_model_cache_formal_20260825_111625_g14c_v5"
                    / "candidate.pt"
                ),
                "checkpoint_sha256": "0" * 64,
            }
        ],
    }
    (tmp_path / "dev_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="v1-v5 invalid runs"):
        checkpoint_freeze(tmp_path, protocol)


def test_readiness_v7_exact_check_set() -> None:
    names = {
        "g14c_v5_failure_registered",
        "producer_consumer_matrix_complete",
        "resolved_context_contract_frozen",
        "outer_nested_expansion_equal",
        "context_negative_cases_pass",
        "legacy_invalid_runs_hard_rejected",
        "clean_worktree_without_local_venv",
        "clean_import_origin",
        "window_reachability_60_of_60",
        "real_preflight_completed",
        "real_tests_phase_completed",
        "phase_and_cell_transactions_regression",
        "portable_fairness_checkpoint_regression",
        "full_pytest_and_smoke_pass",
        "holdout_sealed",
        "no_formal_training_or_performance",
    }
    assert readiness_v7({name: True for name in names}) == READY_V7_VERDICT
    checks = {name: True for name in names}
    checks["real_preflight_completed"] = False
    assert readiness_v7(checks) == "BLOCKED_G14R5_READINESS_V7"
