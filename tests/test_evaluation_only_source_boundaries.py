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
    build_evaluation_execution_contract,
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


def test_evaluation_commands_keep_checkpoint_companions_at_source_run(
    source_reference: dict,
) -> None:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contract = build_evaluation_execution_contract(
        source_reference=source_reference,
        evaluation_run_id="typed_model_cache_evaluation_only_test_pending",
        evaluation_run_root=(
            ROOT.parents[2]
            / "artifacts/experiments/typed_model_cache_evaluation_only"
            / "typed_model_cache_evaluation_only_test_pending"
        ),
        executor_checkout=ROOT,
        executor_commit=head,
        python_executable=ROOT.parents[2] / ".venv/bin/python",
    )
    observed = set()
    for phase in ("formal_cache_policy", "formal_controller"):
        for outer in contract["command_plans"][phase]["commands"]:
            command = outer[outer.index("--command") + 1 :] if "--command" in outer else outer
            for flag in (
                "--seed_checkpoint_manifest_path",
                "--checkpoint_provenance_manifest_path",
            ):
                path = Path(command[command.index(flag) + 1])
                assert path.is_relative_to(SOURCE_RUN_ROOT)
                assert path.is_file()
                observed.add((flag, path.parent.name))
    assert observed == {
        (flag, capacity)
        for flag in (
            "--seed_checkpoint_manifest_path",
            "--checkpoint_provenance_manifest_path",
        )
        for capacity in ("constrained_288mb", "medium_576mb", "relaxed_864mb")
    }
