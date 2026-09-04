from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.runtime.active_formal_bundle import READY_STATUS, validate_active_formal_bundle
from src.runtime.formal_protocol_capabilities import (
    ACTIVE_EXECUTION_PROTOCOL_VERSION,
    FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION,
    FormalProtocolCapabilityError,
    get_protocol_capabilities,
    protocol_capability_matrix,
    require_live_execution_protocol,
)
from src.runtime.formal_training_identity import build_execution_binding
from src.runtime.resolved_formal_execution_context import (
    atomic_create_resolved_formal_execution_context,
    build_resolved_formal_execution_context,
    canonical_sha256,
    validate_resolved_formal_execution_context,
)


ROOT = Path(__file__).resolve().parents[1]
V24_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905"
V24 = V24_ROOT / "protocol_v2_4_manifest.json"
V23 = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903"
    / "protocol_v2_3_manifest.json"
)
ENVIRONMENT = V24_ROOT / "execution_environment_manifest.json"
WINDOW_CONTRACT = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
    / "formal_window_consumption_contract.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip()


def build_persisted_context(tmp_path: Path) -> tuple[dict, dict, Path]:
    protocol = load(V24)
    validate_protocol_v1_1(protocol)
    index = load(V24_ROOT / "protocol_index.json")
    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_ready=index.get("status") == READY_STATUS,
    )
    output_root = tmp_path / "durable-run"
    expansion = resolved_expansion_context(
        protocol,
        protocol_path=str(V24),
        output_root=str(output_root),
        python_executable=sys.executable,
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
        active_protocol_index_path=str(V24_ROOT / "protocol_index.json"),
        active_bundle_resource_resolution_audit_sha256="a" * 64,
    )
    outer = validate_command_templates(
        protocol["execution_contract"]["command_templates"], expansion
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    environment_identity = load(ENVIRONMENT)["scientific_identity"]
    scientific = load(V24_ROOT / "agent_training_scientific_config.json")
    binding = build_execution_binding(
        protocol=protocol,
        scientific_config=scientific,
        execution_commit=commit,
        environment_identity=environment_identity,
        command_matrix_sha256=outer["command_matrix_sha256"],
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
    )
    payload = build_resolved_formal_execution_context(
        protocol=protocol,
        expansion_context=expansion,
        environment_identity=environment_identity,
        runtime_audit={
            "observed_execution_commit": commit,
            "resolved_python_absolute_path": str(Path(sys.executable).absolute()),
            "resolution_source": "explicit_python_executable",
        },
        environment_manifest_path=ENVIRONMENT,
        outer_expansion_sha256=outer["command_matrix_sha256"],
        phase_count=outer["phase_count"],
        command_count=outer["command_count"],
        execution_binding=binding,
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
    )
    context_path = Path(expansion["resolved_execution_context_path"])
    atomic_create_resolved_formal_execution_context(context_path, payload)
    return payload, outer, context_path


def test_capability_registry_is_explicit_active_unique_and_fail_closed() -> None:
    matrix = protocol_capability_matrix()
    assert matrix["formal_protocol_capability_routing_contract_version"] == "1.0.0"
    assert matrix["active_execution_protocol_version"] == "2.4.0"
    assert ACTIVE_EXECUTION_PROTOCOL_VERSION == "2.4.0"
    assert FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION == "1.0.0"
    active = [
        version
        for version, row in matrix["versions"].items()
        if row["live_execution_allowed"]
    ]
    assert active == ["2.4.0"]
    assert get_protocol_capabilities("2.4.0").persisted_resolved_execution_context_required
    assert get_protocol_capabilities("2.4.0").nullable_metric_contract_required
    assert not get_protocol_capabilities("2.4.0").holdout_capability
    assert get_protocol_capabilities("2.3.0").execution_status == "historical_audit_only"
    assert get_protocol_capabilities("1.5.0").persisted_resolved_execution_context_required
    with pytest.raises(FormalProtocolCapabilityError, match="audit-only"):
        require_live_execution_protocol("2.3.0")
    with pytest.raises(FormalProtocolCapabilityError, match="unregistered"):
        get_protocol_capabilities("2.5.0")


def test_v24_protocol_contract_and_command_matrix_identity() -> None:
    protocol = load(V24)
    assert validate_protocol_v1_1(protocol)["status"] == "pass"
    routing = load(V24_ROOT / "formal_protocol_capability_routing_contract.json")
    assert routing["capability_matrix"] == protocol_capability_matrix()
    assert routing["semantic_sha256"] == protocol[
        "formal_protocol_capability_routing_contract"
    ]["semantic_sha256"]
    default = protocol["execution_contract"]["default_expansion_context"]
    with pytest.raises(FormalExecutionError, match="repository root|python_executable"):
        validate_command_templates(protocol["execution_contract"]["command_templates"], default)


def test_v24_real_nested_wrapper_consumes_persisted_context(tmp_path: Path) -> None:
    if not git_clean():
        pytest.skip("requires the final Git-clean candidate")
    payload, outer, context_path = build_persisted_context(tmp_path)
    output = tmp_path / "nested-preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_typed_model_cache_formal_restart.py"),
            "--protocol-path", str(V24),
            "--output-path", str(output),
            "--window-consumption-contract-path", str(WINDOW_CONTRACT),
            "--resolved-execution-context-path", str(context_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    nested = load(output)
    assert nested["resolved_execution_context"]["context_sha256"] == payload[
        "context_sha256"
    ]
    assert nested["resolved_execution_context"]["expansion_equal"] is True
    assert nested["command_expansion"]["command_matrix_sha256"] == outer[
        "command_matrix_sha256"
    ]
    assert nested["command_expansion"]["phase_count"] == 15
    assert nested["command_expansion"]["command_count"] == 186
    assert nested["window_reachability"]["reachable_count"] == 60
    commands = [
        row
        for phase in nested["command_expansion"]["expanded"].values()
        for row in phase["commands"]
    ]
    assert len(nested["command_expansion"]["expanded"]["train"]["commands"]) == 150
    assert all(row[0] == str(Path(sys.executable).absolute()) for row in commands)
    assert all("/ABSOLUTE/" not in token for row in commands for token in row)
    assert all("{" not in token and "}" not in token for row in commands for token in row)


def test_v24_nested_wrapper_missing_context_rejected_by_real_subprocess(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_typed_model_cache_formal_restart.py"),
            "--protocol-path", str(V24),
            "--output-path", str(tmp_path / "must-not-exist.json"),
            "--window-consumption-contract-path", str(WINDOW_CONTRACT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires resolved execution context" in result.stderr


def test_v24_context_tamper_cross_protocol_and_relative_python_rejected(
    tmp_path: Path,
) -> None:
    if not git_clean():
        pytest.skip("requires the final Git-clean candidate")
    payload, _, _ = build_persisted_context(tmp_path)
    sha_tamper = deepcopy(payload)
    sha_tamper["context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        validate_resolved_formal_execution_context(sha_tamper)
    relative = deepcopy(payload)
    relative["runtime_location"]["resolved_python_absolute_path"] = ".venv/bin/python"
    relative["context_sha256"] = canonical_sha256(
        {key: value for key, value in relative.items() if key != "context_sha256"}
    )
    with pytest.raises(ValueError, match="not absolute"):
        validate_resolved_formal_execution_context(relative)
    with pytest.raises(ValueError, match="protocol identity drift"):
        validate_resolved_formal_execution_context(payload, protocol=load(V23))
