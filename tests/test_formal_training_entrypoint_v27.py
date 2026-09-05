from __future__ import annotations

import ast
import inspect
import json
import symtable
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluators.typed_model_cache_formal_execution import validate_protocol_v1_1
from src.runtime.formal_protocol_capabilities import get_protocol_capabilities
from src.runtime.formal_training_contract import (
    FormalTrainingContractError,
    resolve_training_contract,
)
from src.runtime.formal_training_identity import build_execution_binding


ROOT = Path(__file__).resolve().parents[1]
V27_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905"
PROTOCOL_PATH = V27_ROOT / "protocol_v2_7_manifest.json"
SCIENTIFIC_PATH = V27_ROOT / "agent_training_scientific_config.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture()
def identities() -> tuple[dict, dict, dict, dict]:
    protocol = load(PROTOCOL_PATH)
    scientific = load(SCIENTIFIC_PATH)
    assert validate_protocol_v1_1(protocol)["status"] == "pass"
    environment = deepcopy(
        protocol["formal_execution_environment_contract"]["scientific_identity"]
    )
    commit = "a" * 40
    command_hash = "b" * 64
    bundle_hash = "c" * 64
    binding = build_execution_binding(
        protocol=protocol,
        scientific_config=scientific,
        execution_commit=commit,
        environment_identity=environment,
        command_matrix_sha256=command_hash,
        active_formal_bundle_sha256=bundle_hash,
    )
    context = {
        "scientific_identity": {
            "execution_commit": commit,
            "environment_fingerprint": environment["environment_fingerprint"],
            "dependency_fingerprint": environment["dependency_fingerprint"],
            "agent_scientific_config_semantic_sha256": scientific[
                "config_semantic_sha256"
            ],
            "formal_training_execution_binding_sha256": binding[
                "binding_full_sha256"
            ],
            "environment_identity_projection_contract_version": "1.1.0",
            "full_normalized_environment_projection": environment,
            "formal_agent_order_contract_semantic_sha256": protocol[
                "formal_agent_order_contract"
            ]["semantic_sha256"],
            "active_formal_bundle_sha256": bundle_hash,
            "formal_nullable_metric_aggregation_contract_semantic_sha256": protocol[
                "formal_nullable_metric_aggregation_contract"
            ]["semantic_sha256"],
        },
        "command_expansion": {"resolved_command_matrix_sha256": command_hash},
        "context_sha256": "d" * 64,
    }
    return protocol, scientific, binding, context


def resolve(
    protocol: dict,
    scientific: dict,
    binding: dict,
    context: dict,
    *,
    cli_values: dict | None = None,
):
    return resolve_training_contract(
        agent_name="sa_ghmappo",
        profile_defaults={
            "episodes": 1,
            "update_every": 1,
            "batch_size": 1,
            "max_steps": 1,
        },
        cli_values=cli_values or {},
        formal_protocol=protocol,
        scientific_config=scientific,
        execution_binding=binding,
        resolved_execution_context=context,
    )


def test_active_nullable_resolver_success_and_serialization(
    identities: tuple[dict, dict, dict, dict],
) -> None:
    protocol, scientific, binding, context = identities
    resolved = resolve(protocol, scientific, binding, context)
    serialized = resolved.to_dict()
    assert get_protocol_capabilities("2.7.0").nullable_metric_contract_required
    assert serialized["formal_protocol_version"] == "2.7.0"
    assert serialized["episodes"] == 256
    assert serialized["expected_update_count"] == 32
    assert serialized["checkpoint_every_updates"] == 4
    assert serialized["agent_config"]["auxiliary_coef"] == 0.06
    assert serialized[
        "formal_nullable_metric_aggregation_contract_semantic_sha256"
    ] == protocol["formal_nullable_metric_aggregation_contract"]["semantic_sha256"]


@pytest.mark.parametrize("value", [None, "0" * 64])
def test_nullable_hash_missing_or_drift_is_rejected(
    identities: tuple[dict, dict, dict, dict], value: str | None
) -> None:
    protocol, scientific, binding, context = identities
    drift = deepcopy(context)
    if value is None:
        drift["scientific_identity"].pop(
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        )
    else:
        drift["scientific_identity"][
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] = value
    with pytest.raises(FormalTrainingContractError, match="nullable metric contract"):
        resolve(protocol, scientific, binding, drift)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("agent_scientific_config_semantic_sha256", "scientific config identity"),
        ("formal_training_execution_binding_sha256", "execution binding identity"),
        ("active_formal_bundle_sha256", "active_formal_bundle_sha256"),
        ("formal_agent_order_contract_semantic_sha256", "agent order contract identity"),
    ],
)
def test_scientific_binding_and_context_mismatch_are_rejected(
    identities: tuple[dict, dict, dict, dict], field: str, message: str
) -> None:
    protocol, scientific, binding, context = identities
    drift = deepcopy(context)
    drift["scientific_identity"][field] = "0" * 64
    with pytest.raises(FormalTrainingContractError, match=message):
        resolve(protocol, scientific, binding, drift)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episodes", 1),
        ("update_every", 1),
        ("batch_size", 1),
        ("max_steps", 1),
        ("checkpoint_every_updates", 1),
    ],
)
def test_illegal_formal_budget_override_is_rejected(
    identities: tuple[dict, dict, dict, dict], field: str, value: int
) -> None:
    protocol, scientific, binding, context = identities
    with pytest.raises(FormalTrainingContractError, match="formal CLI override rejected"):
        resolve(protocol, scientific, binding, context, cli_values={field: value})


def test_nonformal_early_return_and_historical_fixture_are_not_active_acceptance(
    identities: tuple[dict, dict, dict, dict],
) -> None:
    legacy = resolve_training_contract(
        agent_name="ppo",
        profile_defaults={
            "episodes": 1,
            "update_every": 1,
            "batch_size": 1,
            "max_steps": 1,
        },
        cli_values={},
    )
    assert legacy.formal_protocol_version is None
    assert legacy.formal_nullable_metric_aggregation_contract_semantic_sha256 is None
    historical = load(
        ROOT
        / "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825"
        / "protocol_v1_6_manifest.json"
    )
    assert historical["typed_model_cache_formal_protocol_version"] != "2.7.0"
    assert not get_protocol_capabilities("1.6.0").live_execution_allowed


def test_resolver_has_no_undefined_protocol_reference() -> None:
    from src.runtime import formal_training_contract

    source = inspect.getsource(formal_training_contract)
    module = ast.parse(source)
    resolver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_training_contract"
    )
    assert not [
        node
        for node in ast.walk(resolver)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "protocol"
    ]
    table = symtable.symtable(source, "formal_training_contract.py", "exec")
    resolver_table = next(
        child for child in table.get_children() if child.get_name() == "resolve_training_contract"
    )
    assert "protocol" not in resolver_table.get_globals()


def test_protocol_27_science_is_identical_to_26() -> None:
    previous = load(
        ROOT
        / "configs/experiment/typed_model_cache_formal_protocol_v2_6_20260905"
        / "protocol_v2_6_manifest.json"
    )
    current = load(PROTOCOL_PATH)
    for field in (
        "workload",
        "agent_matrix",
        "seed_plan",
        "training_budget",
        "typed_catalog_and_capacity",
        "endpoints",
        "ablation_and_support",
        "statistics",
        "claim_evidence_map",
        "comparisons",
        "holdout_execution_contract",
    ):
        assert current[field] == previous[field]
    assert (
        current["formal_nullable_metric_aggregation_contract"]
        == previous["formal_nullable_metric_aggregation_contract"]
    )
    assert (
        current["execution_contract"]["command_templates"]
        == previous["execution_contract"]["command_templates"]
    )
