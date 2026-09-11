from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from src.runtime.evaluation_only_execution import (
    DISPOSITION_ELIGIBILITY_PATH,
    SOURCE_RUN_ROOT,
    EvaluationOnlyError,
    _require_clean_code,
    build_model_source_reference,
    canonical_sha256,
    validate_execution_contract,
    validate_model_source_reference,
)


ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT.parent / "g14r20_i_scientific_a6d1fd8"


@pytest.fixture(scope="module")
def source_reference() -> dict:
    return build_model_source_reference(
        disposition_eligibility_path=DISPOSITION_ELIGIBILITY_PATH,
        source_run_root=SOURCE_RUN_ROOT,
        scientific_checkout=SCIENCE,
    )


def _rehash_source(value: dict) -> dict:
    value["source_reference_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "source_reference_sha256"}
    )
    return value


def test_h_disposition_path_and_source_root_are_fixed(tmp_path: Path) -> None:
    fake = tmp_path / "checkpoint_and_partial_result_eligibility.json"
    fake.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluationOnlyError, match="H disposition trust root"):
        build_model_source_reference(
            disposition_eligibility_path=fake,
            source_run_root=SOURCE_RUN_ROOT,
            scientific_checkout=SCIENCE,
        )
    with pytest.raises(EvaluationOnlyError, match="unreviewed model source run"):
        build_model_source_reference(
            disposition_eligibility_path=DISPOSITION_ELIGIBILITY_PATH,
            source_run_root=tmp_path / SOURCE_RUN_ROOT.name,
            scientific_checkout=SCIENCE,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "agent", "wrong_agent"),
        ("model", "seed", 999),
        ("model", "capacity_label", "wrong_capacity"),
        ("model", "checkpoint_sha256", "0" * 64),
        ("source", "source_context_sha256", "0" * 64),
        ("source", "source_context_identity_sha256", "0" * 64),
        ("source", "source_run_id", "wrong-run"),
        ("source", "disposition_eligibility_sha256", "0" * 64),
        ("source", "scientific_git_tree", "0" * 40),
    ],
)
def test_actual_source_reference_mutations_fail_closed(
    source_reference: dict, section: str, field: str, value: object
) -> None:
    changed = deepcopy(source_reference)
    target = changed["models"][0] if section == "model" else changed
    target[field] = value
    _rehash_source(changed)
    with pytest.raises(EvaluationOnlyError):
        validate_model_source_reference(changed)


def test_checkpoint_symlink_reference_fails_closed(
    source_reference: dict, tmp_path: Path
) -> None:
    changed = deepcopy(source_reference)
    link = tmp_path / "checkpoint.pt"
    link.symlink_to(changed["models"][0]["checkpoint_path"])
    changed["models"][0]["checkpoint_path"] = str(link)
    _rehash_source(changed)
    with pytest.raises(EvaluationOnlyError):
        validate_model_source_reference(changed)


def test_live_dirty_checkout_guard(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    tracked = repo / "src" / "module.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", "src/module.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=test", "-c",
            "user.email=test@example.invalid", "commit", "-m", "fixture",
        ],
        check=True,
        capture_output=True,
    )
    _require_clean_code(repo)
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(EvaluationOnlyError, match="not clean"):
        _require_clean_code(repo)


def test_execution_context_executor_cross_binding_is_recomputed() -> None:
    # The integration request exercises live validation; this mutation isolates
    # the contract-to-context edge and cannot be repaired by rehashing only the contract.
    request_path = (
        ROOT.parents[2]
        / "artifacts/analysis/g14r20_i_evaluation_only_readiness_20260911/authorization_request.json"
    )
    if not request_path.is_file():
        pytest.skip("readiness request is an external integration artifact")
    import json

    request = json.loads(request_path.read_text(encoding="utf-8"))
    changed = deepcopy(request["evaluation_execution_contract"])
    changed["executor_commit"] = "0" * 40
    changed["execution_contract_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "execution_contract_sha256"}
    )
    with pytest.raises(EvaluationOnlyError, match="context/executor"):
        validate_execution_contract(
            changed,
            model_source_reference=request["model_source_reference"],
            check_live=False,
        )
