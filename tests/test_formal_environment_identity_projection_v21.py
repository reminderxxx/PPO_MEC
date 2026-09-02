from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.runtime.active_formal_bundle import (
    ActiveFormalBundleError,
    validate_active_formal_bundle,
)
from src.runtime.formal_execution_environment import (
    ENVIRONMENT_FINGERPRINT_FIELD,
    FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION,
    PROTOCOL_BOUND_EXTENSION_FIELDS,
    ExecutionEnvironmentError,
    assert_child_environment_parity,
    build_environment_identity_projection,
    normalize_environment_identity,
    protocol_bound_extensions_from_protocol,
    resolve_execution_environment,
)
from src.runtime.formal_training_identity import build_execution_binding
from src.runtime.formal_agent_order import resolve_formal_agent_order
from src.runtime.formal_training_contract import resolve_training_contract
from src.runtime.resolved_formal_execution_context import (
    build_resolved_formal_execution_context,
)
from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context


ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901"
V20 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831"
V21 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_1_20260831"
PROTOCOL = json.loads((V22 / "protocol_v2_2_manifest.json").read_text())
MANIFEST = json.loads((V22 / "execution_environment_manifest.json").read_text())
IDENTITY = MANIFEST["scientific_identity"]
EXTENSIONS = protocol_bound_extensions_from_protocol(PROTOCOL)
INDEX_PAYLOAD = json.loads((V22 / "protocol_index.json").read_text())
ACTIVE_READY = INDEX_PAYLOAD.get("status") == "READY_FOR_G14C_V12_CLEAN_TRAIN_AND_FORMAL"


def runtime_projection() -> dict:
    return {
        key: deepcopy(value)
        for key, value in IDENTITY.items()
        if key not in {*PROTOCOL_BOUND_EXTENSION_FIELDS, ENVIRONMENT_FINGERPRINT_FIELD}
    }


def resolved():
    return resolve_execution_environment(
        clean_worktree_root=ROOT,
        execution_commit="ignored_by_projection_v1",
        python_executable=sys.executable,
        environment_manifest=MANIFEST,
        expected_identity=IDENTITY,
        protocol_bound_extensions=EXTENSIONS,
    )


def test_01_manifest_builder_and_runtime_resolver_match_full_projection() -> None:
    result = resolved()
    assert result.environment_identity == IDENTITY
    assert result.runtime_audit["full_normalized_environment_projection"] == IDENTITY


@pytest.mark.parametrize("field", PROTOCOL_BOUND_EXTENSION_FIELDS)
def test_02_extension_fields_enter_fingerprint(field: str) -> None:
    changed = deepcopy(EXTENSIONS)
    major, minor, patch = changed[field].split(".")
    changed[field] = f"{major}.{minor}.{int(patch) + 1}"
    with pytest.raises(ExecutionEnvironmentError, match=field):
        build_environment_identity_projection(runtime_projection(), changed)


@pytest.mark.parametrize("field", PROTOCOL_BOUND_EXTENSION_FIELDS)
def test_03_missing_extension_fails_fast(field: str) -> None:
    changed = deepcopy(EXTENSIONS)
    changed.pop(field)
    with pytest.raises(ExecutionEnvironmentError, match="missing"):
        build_environment_identity_projection(runtime_projection(), changed)


def test_04_unknown_extension_fails_fast() -> None:
    changed = {**EXTENSIONS, "unknown_contract_version": "1.0.0"}
    with pytest.raises(ExecutionEnvironmentError, match="unknown"):
        build_environment_identity_projection(runtime_projection(), changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "drift"),
        ("dependency_fingerprint", "0" * 64),
        ("python_version", "0.0.0"),
        ("torch_version", "0.0.0"),
        ("execution_commit", "legacy A11 label"),
        ("source_root_identity", {"project_package": "src", "source_tree_identity_rule": "drift"}),
    ],
)
def test_05_runtime_observable_drift_fails_fast(field: str, value: object) -> None:
    changed = deepcopy(IDENTITY)
    changed[field] = value
    with pytest.raises(ExecutionEnvironmentError):
        normalize_environment_identity(changed)


def test_06_host_paths_do_not_enter_identity() -> None:
    result = resolved()
    encoded = json.dumps(result.environment_identity, sort_keys=True)
    for host_value in (
        result.python_executable,
        str(ROOT),
        str(result.runtime_audit["cwd"]),
        *map(str, result.runtime_audit["site_packages_paths"]),
    ):
        assert host_value not in encoded


def test_07_canonical_key_order_and_json_round_trip_are_stable() -> None:
    first = build_environment_identity_projection(runtime_projection(), EXTENSIONS)
    reverse = dict(reversed(list(runtime_projection().items())))
    second = build_environment_identity_projection(reverse, dict(reversed(list(EXTENSIONS.items()))))
    assert first == second
    assert normalize_environment_identity(json.loads(json.dumps(first))) == first


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_08_non_finite_values_are_rejected(bad: float) -> None:
    changed = runtime_projection()
    changed["installed_package_count"] = bad
    with pytest.raises(ExecutionEnvironmentError):
        build_environment_identity_projection(changed, EXTENSIONS)


def test_09_expected_identity_cannot_inject_arbitrary_field() -> None:
    changed = {**IDENTITY, "silently_injected": "forbidden"}
    with pytest.raises(ExecutionEnvironmentError, match="unknown"):
        resolve_execution_environment(
            clean_worktree_root=ROOT,
            execution_commit="ignored",
            python_executable=sys.executable,
            expected_identity=changed,
            protocol_bound_extensions=EXTENSIONS,
        )


def test_10_builder_resolver_and_child_parity_are_deterministic() -> None:
    first = resolved()
    second = resolved()
    assert first.environment_identity == second.environment_identity
    parity = assert_child_environment_parity(
        first, clean_worktree_root=ROOT, execution_commit="ignored"
    )
    assert parity == {
        "status": "pass",
        "same_python": True,
        "same_environment_fingerprint": True,
    }


def test_11_active_index_manifest_protocol_and_projection_are_atomic() -> None:
    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_ready=ACTIVE_READY,
        require_clean_git=False,
        require_origin_main_match=False,
    )
    assert bundle["environment_manifest"]["scientific_identity"] == IDENTITY
    assert bundle["index"]["environment_identity"]["environment_fingerprint"] == IDENTITY["environment_fingerprint"]
    assert bundle["index"]["environment_identity"]["projection_contract_version"] == FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
    assert bundle["holdout_capability"] is False


def test_11b_unpushed_candidate_is_rejected_by_formal_origin_gate() -> None:
    head = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    origin = __import__("subprocess").check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    if head == origin:
        validate_active_formal_bundle(
            repository_root=ROOT,
            require_ready=ACTIVE_READY,
            require_clean_git=False,
            require_origin_main_match=True,
        )
    else:
        with pytest.raises(ActiveFormalBundleError, match="HEAD == origin/main"):
            validate_active_formal_bundle(
                repository_root=ROOT,
                require_ready=ACTIVE_READY,
                require_clean_git=False,
                require_origin_main_match=True,
            )


def test_12_protocol_20_active_execution_is_rejected() -> None:
    with pytest.raises(ActiveFormalBundleError, match="unique active protocol index"):
        validate_active_formal_bundle(
            repository_root=ROOT,
            index_path=V20 / "protocol_index.json",
            require_clean_git=False,
        )


def test_12b_protocol_21_is_historical_audit_only_and_rejected_as_active() -> None:
    historical = json.loads((V21 / "protocol_v2_1_manifest.json").read_text())
    assert historical["typed_model_cache_formal_protocol_version"] == "2.1.0"
    with pytest.raises(ActiveFormalBundleError, match="unique active protocol index"):
        validate_active_formal_bundle(
            repository_root=ROOT,
            index_path=V21 / "protocol_index.json",
            require_clean_git=False,
        )


def test_13_pre_execution_stop_does_not_fabricate_a_run() -> None:
    stops = PROTOCOL["supersession"]["pre_execution_stops"]
    assert len(stops) == 1
    stop = stops[0]
    assert stop["classification"] == "PRE-EXECUTION STOP / EXECUTION_IDENTITY_MISMATCH"
    assert stop["durable_run_root_created"] is False
    assert stop["phase_ledger_count"] == stop["cell_ledger_count"] == 0
    assert stop["checkpoint_candidate_row_count"] == 0
    assert stop["holdout_opened"] is False
    assert stop["durable_run_root"] is None
    assert all("g14c_v10" not in str(item.get("run_id", "")) for item in PROTOCOL["supersession"]["invalid_execution_runs"])


def test_14_old_small_projection_hash_cannot_impersonate_full_projection() -> None:
    changed = deepcopy(IDENTITY)
    changed["environment_fingerprint"] = "3858b1ba1d25eee329e8601feaaf7136083df3dd3678e00d79496c9e77d176de"
    with pytest.raises(ExecutionEnvironmentError, match="fingerprint"):
        normalize_environment_identity(changed)


def test_15_wrong_types_and_unsupported_major_are_rejected() -> None:
    wrong_type = deepcopy(IDENTITY)
    wrong_type["installed_package_count"] = "31"
    with pytest.raises(ExecutionEnvironmentError, match="wrong type"):
        normalize_environment_identity(wrong_type)
    wrong_major = deepcopy(IDENTITY)
    wrong_major["formal_execution_environment_contract_version"] = "2.0.0"
    with pytest.raises(ExecutionEnvironmentError, match="unsupported"):
        normalize_environment_identity(wrong_major)


def test_16_binding_and_resolved_context_record_the_full_projection(tmp_path: Path) -> None:
    result = resolved()
    index = json.loads((V22 / "protocol_index.json").read_text())
    scientific = json.loads((V22 / "agent_training_scientific_config.json").read_text())
    expansion = resolved_expansion_context(
        PROTOCOL,
        protocol_path=str(V22 / "protocol_v2_2_manifest.json"),
        output_root=str(tmp_path),
        python_executable=sys.executable,
        active_formal_bundle_sha256=index["active_formal_bundle_sha256"],
        active_protocol_index_path=str(V22 / "protocol_index.json"),
        active_bundle_resource_resolution_audit_sha256="a" * 64,
    )
    binding = build_execution_binding(
        protocol=PROTOCOL,
        scientific_config=scientific,
        execution_commit=result.runtime_audit["observed_execution_commit"],
        environment_identity=result.environment_identity,
        command_matrix_sha256="b" * 64,
        active_formal_bundle_sha256=index["active_formal_bundle_sha256"],
    )
    assert binding["environment_identity"]["full_normalized_projection"] == IDENTITY
    context = build_resolved_formal_execution_context(
        protocol=PROTOCOL,
        expansion_context=expansion,
        environment_identity=result.environment_identity,
        runtime_audit=result.runtime_audit,
        environment_manifest_path=V22 / "execution_environment_manifest.json",
        outer_expansion_sha256="b" * 64,
        phase_count=15,
        command_count=186,
        execution_binding=binding,
        active_formal_bundle_sha256=index["active_formal_bundle_sha256"],
    )
    assert context["scientific_identity"]["full_normalized_environment_projection"] == IDENTITY
    training = resolve_training_contract(
        agent_name="ppo",
        profile_defaults={
            "episodes": 1,
            "update_every": 1,
            "batch_size": 1,
            "max_steps": 1,
            "checkpoint_every_updates": 1,
            "agent_config": {},
        },
        cli_values={},
        formal_protocol=PROTOCOL,
        scientific_config=scientific,
        execution_binding=binding,
        resolved_execution_context=context,
    )
    assert training.formal_protocol_version == "2.2.0"
    assert training.full_normalized_environment_projection == IDENTITY
    order = resolve_formal_agent_order(protocol=PROTOCOL, scientific_config=scientific)
    assert len(order["main_benchmark_agent_order"]) == 15
