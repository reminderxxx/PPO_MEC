from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_typed_model_cache_evaluation_only import verify_project_grant
from src.runtime.evaluation_only_execution import (
    PHASES,
    EvaluationOnlyError,
    canonical_sha256,
    reject_legacy_result_path,
    validate_command_matrix_parsers,
    validate_execution_contract,
)


def _contract(tmp_path: Path) -> dict:
    counts = {
        "formal_cache_policy": 3,
        "formal_controller": 3,
        "formal_ablation": 2,
        "formal_support": 11,
        "formal_scalability": 3,
        "formal_statistics": 1,
        "formal_gate": 1,
        "complete_without_holdout": 0,
    }
    plans = {
        phase: {
            "commands": [["/absolute/python", f"/executor/{phase}.py"]] * counts[phase],
            "expected_outputs": [],
            "matrix_contexts": [{} for _ in range(counts[phase])],
            "matrix_cell_count": counts[phase],
        }
        for phase in PHASES
    }
    value = {
        "evaluation_only_execution_contract_version": "1.0.0",
        "evaluation_run_id": "typed_model_cache_evaluation_only_fixture",
        "evaluation_run_root": str(tmp_path / "typed_model_cache_evaluation_only_fixture"),
        "executor_checkout": str(tmp_path / "executor"),
        "executor_commit": "e" * 40,
        "executor_git_tree": "t" * 40,
        "python_executable": "/absolute/python",
        "model_source_reference_sha256": "s" * 64,
        "model_source_run_id": "typed_model_cache_formal_20260906_152847_g14c_v16",
        "phases": list(PHASES),
        "command_plans": plans,
        "command_plan_sha256": canonical_sha256(plans),
        "ledger_path": str(tmp_path / "phase_state.jsonl"),
        "cell_ledger_path": str(tmp_path / "cell_state.jsonl"),
        "lock_root": str(tmp_path / ".evaluation_only_locks"),
        "staging_root": str(tmp_path / ".staging"),
        "legacy_result_exclusions": [str(tmp_path / "old288"), str(tmp_path / "old576")],
        "legacy_results_enter_new_statistics": False,
        "formal_execution_authorized": False,
        "formal_execution_started": False,
        "holdout_capability": False,
        "holdout_opened": False,
        "training_commands": 0,
        "dev_selection_commands": 0,
        "checkpoint_freeze_commands": 0,
        "holdout_commands": 0,
        "evaluation_execution_context": {
            "context_sha256": "c" * 64,
            "evaluation_execution_identity": {
                "model_source_reference_sha256": "s" * 64,
                "evaluation_run_id": "typed_model_cache_evaluation_only_fixture",
                "executor_commit": "e" * 40,
                "executor_git_tree": "t" * 40,
            },
        },
        "evaluation_execution_context_sha256": "c" * 64,
    }
    context = value["evaluation_execution_context"]
    context["context_sha256"] = canonical_sha256(
        {key: item for key, item in context.items() if key != "context_sha256"}
    )
    value["evaluation_execution_context_sha256"] = context["context_sha256"]
    value["execution_contract_sha256"] = canonical_sha256(value)
    return value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("holdout_capability", True),
        ("holdout_commands", 1),
        ("training_commands", 1),
        ("formal_execution_authorized", True),
    ],
)
def test_unsigned_contract_rejects_scope_expansion(
    tmp_path: Path, field: str, value: object
) -> None:
    contract = _contract(tmp_path)
    contract[field] = value
    contract["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in contract.items() if key != "execution_contract_sha256"}
    )
    with pytest.raises(EvaluationOnlyError):
        validate_execution_contract(contract, check_live=False)


def test_command_phase_membership_and_run_identity_are_hash_bound(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    assert validate_execution_contract(contract, check_live=False)["phase_count"] == 8
    changed = deepcopy(contract)
    changed["evaluation_run_id"] = "wrong-run"
    with pytest.raises(EvaluationOnlyError, match="hash mismatch"):
        validate_execution_contract(changed, check_live=False)


def test_command_plan_digest_is_recomputed(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    contract["command_plans"]["formal_controller"]["commands"] = [
        ["/bin/echo", "/tmp/not_the_benchmark.py"]
    ] * 3
    contract["command_plans"]["formal_controller"]["matrix_contexts"] = [{}, {}, {}]
    contract["command_plans"]["formal_controller"]["matrix_cell_count"] = 3
    contract["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in contract.items() if key != "execution_contract_sha256"}
    )
    with pytest.raises(EvaluationOnlyError, match="command plan hash"):
        validate_execution_contract(contract, check_live=False)


def test_execution_contract_is_cross_bound_to_model_source(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    source = {
        "source_run_id": contract["model_source_run_id"],
        "legacy_result_exclusions": contract["legacy_result_exclusions"],
    }
    source["source_reference_sha256"] = canonical_sha256(source)
    contract["model_source_reference_sha256"] = source["source_reference_sha256"]
    contract["evaluation_execution_context"]["evaluation_execution_identity"][
        "model_source_reference_sha256"
    ] = source["source_reference_sha256"]
    context = contract["evaluation_execution_context"]
    context["context_sha256"] = canonical_sha256(
        {key: item for key, item in context.items() if key != "context_sha256"}
    )
    contract["evaluation_execution_context_sha256"] = context["context_sha256"]
    contract["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in contract.items() if key != "execution_contract_sha256"}
    )
    assert validate_execution_contract(
        contract, model_source_reference=source, check_live=False
    )["phase_count"] == 8
    changed = deepcopy(contract)
    changed["model_source_run_id"] = "wrong-run"
    changed["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "execution_contract_sha256"}
    )
    with pytest.raises(EvaluationOnlyError, match="model-source run"):
        validate_execution_contract(changed, model_source_reference=source, check_live=False)


def test_command_parser_rejects_interpreter_and_entrypoint_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    contract["executor_checkout"] = str(tmp_path)
    contract["python_executable"] = "/bin/echo"
    plan = contract["command_plans"]["formal_controller"]
    plan["commands"] = [["/bin/echo", "/tmp/not_the_benchmark.py"]] * 3
    plan["matrix_contexts"] = [{}, {}, {}]
    plan["matrix_cell_count"] = 3
    contract["command_plan_sha256"] = canonical_sha256(contract["command_plans"])
    contract["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in contract.items() if key != "execution_contract_sha256"}
    )
    with pytest.raises(EvaluationOnlyError, match="entrypoint drift"):
        validate_command_matrix_parsers(contract, check_live=False)
    changed = deepcopy(contract)
    changed["phases"] = [*PHASES, "sealed_holdout"]
    changed["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "execution_contract_sha256"}
    )
    with pytest.raises(EvaluationOnlyError, match="phase authority"):
        validate_execution_contract(changed, check_live=False)


@pytest.mark.parametrize("name", ["old288/result.csv", "old576/staging.json"])
def test_legacy_partial_and_failed_staging_cannot_enter_statistics(
    tmp_path: Path, name: str
) -> None:
    contract = _contract(tmp_path)
    with pytest.raises(EvaluationOnlyError, match="excluded from new statistics"):
        reject_legacy_result_path(tmp_path / name, contract)


def test_missing_or_false_project_grant_cannot_create_run(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    package = {
        "authorization_request_sha256": "a" * 64,
        "evaluation_execution_contract": contract,
    }
    grant = {
        "version": "1.0.0",
        "status": "READY_FOR_EVALUATION_ONLY_AUTHORIZATION",
        "authorization_request_sha256": package["authorization_request_sha256"],
        "evaluation_run_id": contract["evaluation_run_id"],
        "model_source_reference_sha256": contract["model_source_reference_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "executor_commit": contract["executor_commit"],
        "phases": list(PHASES),
        "holdout_capability": False,
        "independent_review": {},
        "issued_at": "2026-09-11T00:00:00+00:00",
        "expires_at": "2026-09-12T00:00:00+00:00",
    }
    with pytest.raises(EvaluationOnlyError):
        verify_project_grant(package, grant)
    assert not Path(contract["evaluation_run_root"]).exists()
