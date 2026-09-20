"""G14R20-I5-E consumer boundary and preserved real non-formal acceptance."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_typed_model_cache_formal_support import build_parser
from scripts.run_typed_model_cache_restricted_recovery import preflight_recovery_scientific_inputs
from src.evaluators.formal_cell_transaction import validate_producer_integrity_manifests
from src.runtime.generated_checkpoint_resources import (
    GeneratedCheckpointResourceError,
    evaluation_model_source_scope,
)
from src.runtime.restricted_recovery import (
    ALLOWED_CELL_IDS,
    validate_recovery_handoff_manifest,
)


ROOT = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_restricted_recovery"
    "/nonformal_i5e_consumer_acceptance_20260920_f"
)


@pytest.fixture
def accepted():
    path = ROOT / "restricted_recovery_request.json"
    if not path.is_file():
        pytest.skip("preserved I5-E non-formal acceptance artifact is unavailable")
    request = json.loads(path.read_text())
    command = request["recovery_execution"]["command_plan"]["commands"][0]
    args = build_parser().parse_args(command[2:])
    protocol = json.loads(Path(args.protocol_path).read_text())
    return request, args, protocol


def test_real_nonformal_two_process_publication_readback(accepted):
    request, args, protocol = accepted
    assert evaluation_model_source_scope(args, default_run_root=ROOT, protocol=protocol)["evaluation_only"]
    records = [json.loads(line) for line in (ROOT / "cell_state.jsonl").read_text().splitlines()]
    assert [(row["cell_id"], row["attempt"]) for row in records if row["status"] == "committed"] == [
        (ALLOWED_CELL_IDS[0], 3), (ALLOWED_CELL_IDS[1], 1)
    ]
    for path in sorted((ROOT / "formal_ablation").iterdir()):
        assert validate_producer_integrity_manifests(path)["manifest_count"] == 1
        summaries = list(path.rglob("*.summary.json"))
        assert len(summaries) == 1
        summary = json.loads(summaries[0].read_text())
        assert summary["run_info"]["checkpoint_provenance_status"] == "compatible"
        assert len(summary["cache_event_trace"]) > 0
        manifest = json.loads((path / "run_manifest.json").read_text())
        assert manifest["non_formal_rehearsal"] is True
        assert manifest["formal_performance_evidence"] is False
        assert manifest["observed_episode_count"] == 1
    handoff = json.loads((ROOT / "nonformal_acceptance_handoff.json").read_text())
    assert validate_recovery_handoff_manifest(handoff)["unique_cell_count"] == 8


@pytest.mark.parametrize("mutation", [
    "wrong_run", "wrong_context", "wrong_executor", "wrong_source",
    "wrong_phase", "missing_contract", "contract_conflict", "symlink_reference",
])
def test_recovery_scope_rejects_drift_before_rollout(accepted, monkeypatch, mutation):
    request, args, protocol = accepted
    import src.runtime.generated_checkpoint_resources as resources

    original_read = resources._strict_json_object
    original_exists = Path.exists
    original_symlink = Path.is_symlink

    def read_mutated(path, label):
        value = original_read(path, label)
        value = deepcopy(value)
        if mutation == "wrong_run" and path.name == "resolved_execution_context.json":
            value["evaluation_execution_identity"]["evaluation_run_id"] = "other"
        elif mutation == "wrong_context" and path.name == "resolved_execution_context.json":
            value["context_sha256"] = "0" * 64
        elif mutation == "wrong_executor" and path.name == "resolved_execution_context.json":
            value["evaluation_execution_identity"]["executor_commit"] = "0" * 40
        elif mutation == "wrong_source" and path.name == "evaluation_model_source_reference.json":
            value["source_reference_sha256"] = "0" * 64
        elif mutation == "wrong_phase" and path.name == "restricted_recovery_execution_contract.json":
            value["allowed_phase"] = "formal_support"
        return value

    def exists_mutated(path):
        if mutation == "missing_contract" and path.name == "restricted_recovery_execution_contract.json":
            return False
        if mutation == "contract_conflict" and path.name == "evaluation_execution_contract.json":
            return True
        return original_exists(path)

    def symlink_mutated(path):
        return mutation == "symlink_reference" and path.name == "evaluation_model_source_reference.json" or original_symlink(path)

    monkeypatch.setattr(resources, "_strict_json_object", read_mutated)
    monkeypatch.setattr(Path, "exists", exists_mutated)
    monkeypatch.setattr(Path, "is_symlink", symlink_mutated)
    with pytest.raises((GeneratedCheckpointResourceError, ValueError)):
        evaluation_model_source_scope(args, default_run_root=ROOT, protocol=protocol)


def test_required_scientific_file_is_checked_before_child(accepted):
    request, _, _ = accepted
    request = deepcopy(request)
    command = request["recovery_execution"]["command_plan"]["commands"][0]
    index = command.index("--seed-checkpoint-manifest-path") + 1
    command[index] = "/missing/seed_checkpoint_manifest.json"
    with pytest.raises(ValueError, match="required file missing"):
        preflight_recovery_scientific_inputs(request, command, ALLOWED_CELL_IDS[0])


def test_json_contract_rejects_symlinked_parent(tmp_path: Path):
    from src.runtime.generated_checkpoint_resources import _strict_json_object

    real = tmp_path / "real"
    real.mkdir()
    (real / "contract.json").write_text("{}\n")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(GeneratedCheckpointResourceError, match="symlink"):
        _strict_json_object(tmp_path / "alias/contract.json", "contract")
