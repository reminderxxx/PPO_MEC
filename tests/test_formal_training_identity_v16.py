from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_typed_model_cache_formal_protocol import (
    reject_invalid_run_root,
    resolved_expansion_context,
)
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    READY_V8_VERDICT,
    readiness_v8,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
from src.runtime.formal_training_contract import (
    FormalTrainingContractError,
    resolve_training_contract,
)
from src.runtime.formal_training_identity import (
    FormalTrainingIdentityError,
    build_execution_binding,
    canonical_sha256,
    load_strict_json_mapping,
    scientific_config_projection,
    validate_checkpoint_training_identity,
    validate_execution_binding,
    validate_scientific_config,
)


ROOT = Path(__file__).resolve().parents[1]
V13_CONFIG = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821/agent_training_configs.json"
V15 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_5_20260825/protocol_v1_5_manifest.json"
V16_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825"
V16 = V16_ROOT / "protocol_v1_6_manifest.json"
SCIENTIFIC = V16_ROOT / "agent_training_scientific_config.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture()
def protocol() -> dict:
    payload = load(V16)
    assert validate_protocol_v1_1(payload)["status"] == "pass"
    return payload


@pytest.fixture()
def scientific(protocol: dict) -> dict:
    payload = load_strict_json_mapping(SCIENTIFIC, "agent scientific config")
    assert validate_scientific_config(payload, protocol=protocol)["agent_count"] == 10
    return payload


@pytest.fixture()
def runtime_identity(protocol: dict) -> tuple[dict, str, str]:
    context = resolved_expansion_context(
        protocol,
        protocol_path=str(V16),
        output_root="/tmp/g14r6-binding-test",
        python_executable="/usr/bin/python3",
    )
    command = validate_command_templates(
        protocol["execution_contract"]["command_templates"], context
    )
    environment = protocol["formal_execution_environment_contract"]["scientific_identity"]
    return environment, "a" * 40, command["command_matrix_sha256"]


@pytest.fixture()
def binding(protocol: dict, scientific: dict, runtime_identity: tuple[dict, str, str]) -> dict:
    environment, commit, command_hash = runtime_identity
    return build_execution_binding(
        protocol=protocol,
        scientific_config=scientific,
        execution_commit=commit,
        environment_identity=environment,
        command_matrix_sha256=command_hash,
    )


def rehash_scientific(payload: dict) -> dict:
    result = deepcopy(payload)
    result["config_semantic_sha256"] = canonical_sha256(
        scientific_config_projection(result)
    )
    return result


def test_g14c_v6_legacy_companion_failure_is_exactly_reproduced() -> None:
    with pytest.raises(
        FormalTrainingContractError,
        match="agent config companion protocol hash mismatch",
    ):
        resolve_training_contract(
            agent_name="sa_ghmappo",
            profile_defaults={"episodes": 1, "update_every": 1, "batch_size": 1, "max_steps": 1},
            cli_values={},
            formal_protocol=load(V15),
            agent_config_companion=load(V13_CONFIG),
        )


def test_scientific_config_and_current_binding_resolve_successfully(
    protocol: dict,
    scientific: dict,
    binding: dict,
    runtime_identity: tuple[dict, str, str],
) -> None:
    environment, commit, command_hash = runtime_identity
    context = {
        "scientific_identity": {
            "execution_commit": commit,
            "environment_fingerprint": environment["environment_fingerprint"],
            "dependency_fingerprint": environment["dependency_fingerprint"],
            "agent_scientific_config_semantic_sha256": scientific["config_semantic_sha256"],
            "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
        },
        "command_expansion": {"resolved_command_matrix_sha256": command_hash},
        "context_sha256": "b" * 64,
    }
    resolved = resolve_training_contract(
        agent_name="sa_ghmappo",
        profile_defaults={"episodes": 1, "update_every": 1, "batch_size": 1, "max_steps": 1},
        cli_values={},
        formal_protocol=protocol,
        scientific_config=scientific,
        execution_binding=binding,
        resolved_execution_context=context,
    )
    assert resolved.contract_version == "2.0.0"
    assert resolved.agent_config["auxiliary_coef"] == 0.06
    assert resolved.formal_training_execution_binding_sha256 == binding["binding_full_sha256"]


def test_binding_missing_and_legacy_companion_rejected(protocol: dict, scientific: dict) -> None:
    with pytest.raises(FormalTrainingContractError, match="execution_binding_path"):
        resolve_training_contract(
            agent_name="ppo",
            profile_defaults={"episodes": 1, "update_every": 1, "batch_size": 1, "max_steps": 1},
            cli_values={}, formal_protocol=protocol, scientific_config=scientific,
            resolved_execution_context={},
        )
    with pytest.raises(FormalTrainingContractError, match="rejects legacy"):
        resolve_training_contract(
            agent_name="ppo",
            profile_defaults={"episodes": 1, "update_every": 1, "batch_size": 1, "max_steps": 1},
            cli_values={}, formal_protocol=protocol, agent_config_companion=load(V13_CONFIG),
            scientific_config=scientific, execution_binding={}, resolved_execution_context={},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("config_semantic_sha256", "0" * 64), "semantic SHA-256"),
        (lambda value: value["agents"]["ppo"]["hyperparameters"].__setitem__("learning_rate", 0.9), "semantic SHA-256"),
        (lambda value: value["agents"]["sa_ghmappo"]["hyperparameters"].__setitem__("auxiliary_coef", 0.05), "auxiliary_coef drift"),
        (lambda value: value["agents"].pop("dqn"), "missing, duplicate, or unknown"),
        (lambda value: value["agents"].__setitem__("unknown", deepcopy(value["agents"]["dqn"])), "missing, duplicate, or unknown"),
        (lambda value: value.__setitem__("unknown", True), "unknown top-level"),
        (lambda value: value.__setitem__("canonical_serialization", "implementation-defined"), "canonical serialization drift"),
        (lambda value: value["agents"]["ppo"]["hyperparameters"].__setitem__("learning_rate", float("nan")), "non-finite"),
    ],
)
def test_scientific_config_negative_cases(
    scientific: dict, mutation, match: str
) -> None:
    drift = deepcopy(scientific)
    mutation(drift)
    with pytest.raises(FormalTrainingIdentityError, match=match):
        validate_scientific_config(drift)


def test_rehashed_learning_rate_and_auxiliary_drift_rejected_by_protocol(
    protocol: dict, scientific: dict
) -> None:
    for agent, field, value in (
        ("ppo", "learning_rate", 0.9),
        ("sa_ghmappo", "auxiliary_coef", 0.05),
    ):
        drift = deepcopy(scientific)
        drift["agents"][agent]["hyperparameters"][field] = value
        drift = rehash_scientific(drift)
        with pytest.raises(FormalTrainingIdentityError, match="hyperparameter mismatch|auxiliary_coef drift|protocol scientific config hash mismatch"):
            validate_scientific_config(drift, protocol=protocol)


@pytest.mark.parametrize(
    "field",
    [
        "protocol_identity",
        "execution_commit",
        "agent_scientific_config_semantic_sha256",
        "environment_identity",
        "data_and_runtime_identity",
        "command_matrix_sha256",
        "portable_resource_identity",
    ],
)
def test_execution_binding_drift_rejected(
    protocol: dict, scientific: dict, binding: dict,
    runtime_identity: tuple[dict, str, str], field: str,
) -> None:
    environment, commit, command_hash = runtime_identity
    drift = deepcopy(binding)
    drift[field] = "0" * 64 if isinstance(drift[field], str) else {}
    with pytest.raises(FormalTrainingIdentityError, match="full SHA-256|execution binding drift"):
        validate_execution_binding(
            drift, protocol=protocol, scientific_config=scientific,
            execution_commit=commit, environment_identity=environment,
            command_matrix_sha256=command_hash,
        )


def test_cross_pair_and_protocol_semantic_drift_rejected(
    protocol: dict, scientific: dict, binding: dict,
    runtime_identity: tuple[dict, str, str],
) -> None:
    environment, commit, command_hash = runtime_identity
    changed = deepcopy(scientific)
    changed["agents"]["ppo"]["hyperparameters"]["learning_rate"] = 0.9
    changed = rehash_scientific(changed)
    with pytest.raises(FormalTrainingIdentityError):
        validate_execution_binding(
            binding, protocol=protocol, scientific_config=changed,
            execution_commit=commit, environment_identity=environment,
            command_matrix_sha256=command_hash,
        )
    changed_protocol = deepcopy(protocol)
    changed_protocol["status"] = "drift"
    changed_protocol = attach_hashes(changed_protocol)
    with pytest.raises(FormalTrainingIdentityError):
        validate_execution_binding(
            binding, protocol=changed_protocol, scientific_config=scientific,
            execution_commit=commit, environment_identity=environment,
            command_matrix_sha256=command_hash,
        )


def test_content_identical_path_relocation_allowed_and_same_name_drift_rejected(
    scientific: dict, tmp_path: Path
) -> None:
    relocated = tmp_path / "renamed.json"
    relocated.write_text(json.dumps(scientific), encoding="utf-8")
    assert load_strict_json_mapping(relocated, "relocated")["config_semantic_sha256"] == scientific["config_semantic_sha256"]
    drift = deepcopy(scientific)
    drift["config_semantic_sha256"] = "0" * 64
    relocated.write_text(json.dumps(drift), encoding="utf-8")
    with pytest.raises(FormalTrainingIdentityError, match="semantic SHA-256"):
        validate_scientific_config(load_strict_json_mapping(relocated, "drift"))


def test_duplicate_json_key_rejected(tmp_path: Path) -> None:
    target = tmp_path / "duplicate.json"
    target.write_text('{"agents": {}, "agents": {}}', encoding="utf-8")
    with pytest.raises(FormalTrainingIdentityError, match="duplicate JSON key"):
        load_strict_json_mapping(target, "duplicate")


def test_checkpoint_provenance_missing_or_drift_rejected(
    scientific: dict, binding: dict
) -> None:
    expected = {
        "agent_scientific_config_semantic_sha256": scientific["config_semantic_sha256"],
        "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
        "formal_protocol_semantic_sha256": "c" * 64,
        "execution_commit": "a" * 40,
        "resolved_execution_context_sha256": "b" * 64,
    }
    assert validate_checkpoint_training_identity(
        expected,
        scientific_config_sha256=expected["agent_scientific_config_semantic_sha256"],
        binding_sha256=expected["formal_training_execution_binding_sha256"],
        protocol_semantic_sha256=expected["formal_protocol_semantic_sha256"],
        execution_commit=expected["execution_commit"],
        resolved_context_sha256=expected["resolved_execution_context_sha256"],
    )["status"] == "pass"
    for field in expected:
        drift = dict(expected)
        drift.pop(field)
        with pytest.raises(FormalTrainingIdentityError, match=field):
            validate_checkpoint_training_identity(
                drift,
                scientific_config_sha256=expected["agent_scientific_config_semantic_sha256"],
                binding_sha256=expected["formal_training_execution_binding_sha256"],
                protocol_semantic_sha256=expected["formal_protocol_semantic_sha256"],
                execution_commit=expected["execution_commit"],
                resolved_context_sha256=expected["resolved_execution_context_sha256"],
            )


def test_protocol_v16_matrix_and_invalid_v6_boundary(protocol: dict) -> None:
    train = protocol["execution_contract"]["command_templates"]["train"]
    assert len(train["matrix_contexts"]) == 150
    assert "--agent_config_path" not in train["argv"]
    assert len(protocol["supersession"]["invalid_execution_runs"]) == 7
    invalid = protocol["supersession"]["invalid_execution_runs"][-1]
    assert invalid["run_id"].endswith("g14c_v6")
    assert invalid["episode_count"] == invalid["checkpoint_count"] == 0
    for mode in ("fresh", "resume", "finalize"):
        with pytest.raises(FormalExecutionError, match="permanently rejected"):
            reject_invalid_run_root(protocol, ROOT / "artifacts" / invalid["run_id"])


def test_readiness_v8_exact_gate() -> None:
    names = {
        "g14c_v6_failure_registered", "producer_consumer_matrix_complete",
        "scientific_config_contract_frozen", "execution_binding_contract_frozen",
        "ten_agent_config_parity", "training_commands_150_bound",
        "ten_agent_entrypoint_rehearsal", "negative_validation_complete",
        "checkpoint_provenance_consumers_bound", "outer_nested_expansion_equal",
        "clean_worktree_without_local_venv", "clean_import_origin",
        "window_reachability_60_of_60", "real_preflight_completed",
        "real_tests_phase_completed", "phase_cell_resume_finalize_regression",
        "full_pytest_and_smoke_pass", "holdout_sealed",
        "no_formal_training_checkpoint_or_performance",
    }
    assert readiness_v8({name: True for name in names}) == READY_V8_VERDICT
    checks = {name: True for name in names}
    checks["ten_agent_entrypoint_rehearsal"] = False
    assert readiness_v8(checks) == "BLOCKED_G14R6_READINESS_V8"
