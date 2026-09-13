"""Real two-checkout acceptance for evaluation-only active-bundle resolution.

The public support consumer is exercised only with ``--dry-run``.  The fixture
creates identity JSON below pytest's temporary directory; it creates no ledger,
lock, grant, staging tree, rollout, training output, or formal result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.runtime.active_formal_bundle import (
    ActiveFormalBundleError,
    resolve_active_bundle_resource,
    validate_active_formal_bundle,
)
from src.runtime.evaluation_only_execution import (
    EvaluationOnlyError,
    build_evaluation_execution_contract,
    canonical_sha256,
    resolve_scientific_bundle_root,
)
from tests.test_evaluation_only_source_boundaries import source_reference


ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT.parent / "g14r20_i_scientific_a6d1fd8"


@pytest.fixture
def cross_checkout_contract(tmp_path: Path, source_reference: dict) -> tuple[Path, dict]:
    run_root = tmp_path / "g14r20_i4_nonformal_acceptance"
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    contract = build_evaluation_execution_contract(
        source_reference=source_reference,
        evaluation_run_id=run_root.name,
        evaluation_run_root=run_root,
        executor_checkout=ROOT,
        executor_commit=commit,
        python_executable=sys.executable,
    )
    run_root.mkdir()
    for name, payload in (
        ("evaluation_model_source_reference.json", source_reference),
        ("evaluation_execution_contract.json", contract),
        ("resolved_execution_context.json", contract["evaluation_execution_context"]),
    ):
        (run_root / name).write_text(json.dumps(payload), encoding="utf-8")
    return run_root, contract


def test_scientific_bundle_root_is_distinct_from_executor(
    cross_checkout_contract: tuple[Path, dict], source_reference: dict
) -> None:
    _, contract = cross_checkout_contract
    context = contract["evaluation_execution_context"]
    resolved = resolve_scientific_bundle_root(
        source_reference=source_reference,
        evaluation_context=context,
        executor_checkout=ROOT,
    )
    assert resolved == SCIENCE.resolve()
    assert resolved != ROOT.resolve()
    assert context["runtime_location"]["repository_root"] == str(ROOT.resolve())


@pytest.mark.parametrize(
    "phase", ["formal_ablation", "formal_support", "formal_scalability"]
)
def test_actual_public_support_consumer_accepts_real_cross_checkout_dry_run(
    cross_checkout_contract: tuple[Path, dict], phase: str
) -> None:
    run_root, contract = cross_checkout_contract
    before = sorted(path.relative_to(run_root) for path in run_root.rglob("*"))
    command = [*contract["command_plans"][phase]["commands"][0], "--dry-run"]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run_pass"
    assert payload["writes_performed"] is False
    assert sorted(path.relative_to(run_root) for path in run_root.rglob("*")) == before
    assert not (run_root / "phase_state.jsonl").exists()
    assert not (run_root / "cell_state.jsonl").exists()


@pytest.mark.parametrize("mutation", ["source", "commit", "hash"])
def test_source_reference_drift_still_fails_closed(
    cross_checkout_contract: tuple[Path, dict],
    source_reference: dict,
    mutation: str,
) -> None:
    _, contract = cross_checkout_contract
    changed = deepcopy(source_reference)
    if mutation == "source":
        changed["source_run_id"] = "unreviewed-source"
    elif mutation == "commit":
        changed["scientific_commit"] = "0" * 40
    else:
        changed["source_reference_sha256"] = "0" * 64
    if mutation != "hash":
        changed["source_reference_sha256"] = canonical_sha256(
            {key: value for key, value in changed.items() if key != "source_reference_sha256"}
        )
    with pytest.raises(EvaluationOnlyError):
        resolve_scientific_bundle_root(
            source_reference=changed,
            evaluation_context=contract["evaluation_execution_context"],
            executor_checkout=ROOT,
        )


def test_executor_identity_drift_still_fails_closed(
    cross_checkout_contract: tuple[Path, dict], source_reference: dict
) -> None:
    _, contract = cross_checkout_contract
    context = deepcopy(contract["evaluation_execution_context"])
    context["evaluation_execution_identity"]["executor_commit"] = "0" * 40
    with pytest.raises(EvaluationOnlyError, match="executor context identity"):
        resolve_scientific_bundle_root(
            source_reference=source_reference,
            evaluation_context=context,
            executor_checkout=ROOT,
        )


def test_real_bundle_role_index_and_path_drift_still_fail_closed(
    cross_checkout_contract: tuple[Path, dict], source_reference: dict
) -> None:
    _, contract = cross_checkout_contract
    bundle = validate_active_formal_bundle(
        repository_root=SCIENCE,
        require_clean_git=False,
        require_origin_main_match=False,
    )
    with pytest.raises(ActiveFormalBundleError, match="role mismatch"):
        resolve_active_bundle_resource(
            bundle, "protocol_manifest", expected_role="wrong scientific role"
        )
    wrong_index = (
        SCIENCE
        / "configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906"
        / "protocol_index.json"
    )
    with pytest.raises(ActiveFormalBundleError, match="unique active protocol index"):
        validate_active_formal_bundle(
            repository_root=SCIENCE,
            index_path=wrong_index,
            require_clean_git=False,
            require_origin_main_match=False,
        )
    context = deepcopy(contract["evaluation_execution_context"])
    context["resolved_expansion_context"]["active_protocol_index_path"] = str(
        wrong_index
    )
    with pytest.raises(EvaluationOnlyError, match="index path identity drift"):
        resolve_scientific_bundle_root(
            source_reference=source_reference,
            evaluation_context=context,
            executor_checkout=ROOT,
        )
    changed = deepcopy(source_reference)
    changed["protocol_path"] = str(ROOT / "README.md")
    changed["source_reference_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "source_reference_sha256"}
    )
    with pytest.raises(EvaluationOnlyError):
        resolve_scientific_bundle_root(
            source_reference=changed,
            evaluation_context=contract["evaluation_execution_context"],
            executor_checkout=ROOT,
        )
