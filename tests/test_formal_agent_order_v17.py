from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.manage_typed_model_cache_formal_artifacts import (
    checkpoint_freeze,
    dev_select,
)
from scripts.run_typed_model_cache_formal_protocol import reject_invalid_run_root
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    READY_V9_VERDICT,
    readiness_v9,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import attach_hashes, sha256_file
from src.runtime.formal_agent_order import (
    FormalAgentOrderError,
    canonical_sha256,
    contract_projection,
    load_formal_agent_order_contract,
    reject_permanently_invalid_run_references,
    resolve_formal_agent_order,
    validate_formal_agent_order_contract,
)


ROOT = Path(__file__).resolve().parents[1]
V17_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
PROTOCOL_PATH = V17_ROOT / "protocol_v1_7_manifest.json"
SCIENTIFIC_PATH = V17_ROOT / "agent_training_scientific_config.json"
ORDER_PATH = V17_ROOT / "formal_agent_order_contract.json"
V7_RUN_ID = "typed_model_cache_formal_20260826_233222_g14c_v7"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture()
def protocol() -> dict:
    payload = load(PROTOCOL_PATH)
    assert validate_protocol_v1_1(payload)["status"] == "pass"
    return payload


@pytest.fixture()
def scientific() -> dict:
    return load(SCIENTIFIC_PATH)


@pytest.fixture()
def contract() -> dict:
    return load_formal_agent_order_contract(ORDER_PATH)


def fairness_payload(contract: dict, *, controller_order: list[str] | None = None) -> dict:
    return {
        "cache_contract": {
            "typed_model_cache": {
                "controller_agents": list(
                    controller_order or contract["learned_agent_order"]
                )
            }
        },
        "baseline_matrix": [
            {"agent_identity": {"name": name}}
            for name in contract["reactive_agent_order"]
        ],
    }


def test_g14c_v7_mapping_order_failure_is_exactly_reproduced_and_resolved(
    protocol: dict, contract: dict
) -> None:
    observed_mapping_order = list(protocol["training_budget"]["agent_configs"])
    assert observed_mapping_order[:2] == ["cache_offload_drl", "controller_mat"]
    assert observed_mapping_order != contract["learned_agent_order"]
    audit = resolve_formal_agent_order(protocol=protocol)
    assert audit["learned_agent_order"] == contract["learned_agent_order"]


def test_same_set_wrong_order_is_rejected(contract: dict) -> None:
    wrong = list(contract["learned_agent_order"])
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(FormalAgentOrderError, match="correct set but wrong order"):
        resolve_formal_agent_order(
            contract=contract,
            fairness_manifests=[fairness_payload(contract, controller_order=wrong)],
        )


def test_alphabetical_mapping_reserialization_does_not_change_resolver(
    protocol: dict, scientific: dict, contract: dict
) -> None:
    reordered = deepcopy(protocol)
    configs = reordered["training_budget"]["agent_configs"]
    names = list(configs)
    random.Random(41).shuffle(names)
    reordered["training_budget"]["agent_configs"] = {
        name: configs[name] for name in names
    }
    assert resolve_formal_agent_order(
        contract=contract, protocol=reordered, scientific_config=scientific
    )["learned_agent_order"] == contract["learned_agent_order"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda values: values.pop(), "membership drift"),
        (lambda values: values.__setitem__(1, values[0]), "duplicate"),
        (lambda values: values.append("unknown_agent"), "membership drift"),
        (lambda values: values.__setitem__(0, "reactive_lru"), "membership drift"),
    ],
)
def test_missing_duplicate_extra_unknown_and_role_swap_rejected(
    contract: dict, mutation, match: str
) -> None:
    observed = list(contract["learned_agent_order"])
    mutation(observed)
    with pytest.raises(FormalAgentOrderError, match=match):
        resolve_formal_agent_order(
            contract=contract,
            fairness_manifests=[fairness_payload(contract, controller_order=observed)],
        )


def test_popularity_report_only_cannot_enter_main_order(contract: dict) -> None:
    drift = deepcopy(contract)
    drift["main_benchmark_agent_order"].append("popularity_cache_heuristic")
    drift["semantic_sha256"] = canonical_sha256(contract_projection(drift))
    with pytest.raises(FormalAgentOrderError, match="exactly 15|popularity"):
        validate_formal_agent_order_contract(drift)


def test_scientific_fairness_and_template_order_drift_rejected(
    protocol: dict, scientific: dict, contract: dict
) -> None:
    drift_scientific = deepcopy(scientific)
    drift_scientific["learned_agent_order"][0:2] = reversed(
        drift_scientific["learned_agent_order"][0:2]
    )
    with pytest.raises(FormalAgentOrderError, match="wrong order"):
        resolve_formal_agent_order(
            contract=contract,
            protocol=protocol,
            scientific_config=drift_scientific,
        )
    wrong_controller = list(contract["learned_agent_order"])
    wrong_controller[-2:] = reversed(wrong_controller[-2:])
    with pytest.raises(FormalAgentOrderError, match="wrong order"):
        resolve_formal_agent_order(
            contract=contract,
            fairness_manifests=[
                fairness_payload(contract, controller_order=wrong_controller)
            ],
        )
    drift_protocol = deepcopy(protocol)
    argv = drift_protocol["execution_contract"]["command_templates"][
        "formal_controller"
    ]["argv"]
    start = argv.index("--agents") + 1
    argv[start], argv[start + 1] = argv[start + 1], argv[start]
    with pytest.raises(FormalAgentOrderError, match="wrong order"):
        resolve_formal_agent_order(contract=contract, protocol=drift_protocol)


def test_dev_selector_source_cannot_bypass_resolver() -> None:
    source = (ROOT / "scripts/run_typed_model_cache_formal_dev_selection.py").read_text(
        encoding="utf-8"
    )
    assert "resolve_formal_agent_order" in source
    assert 'list(protocol["training_budget"]["agent_configs"])' not in source
    from src.runtime.formal_protocol_capabilities import get_protocol_capabilities

    capabilities = get_protocol_capabilities("1.7.0")
    assert capabilities.persisted_resolved_execution_context_required
    assert capabilities.agent_order_contract_required
    assert not capabilities.live_execution_allowed
    preflight_source = (
        ROOT / "scripts/validate_typed_model_cache_formal_restart.py"
    ).read_text(encoding="utf-8")
    assert "load_resolved_formal_execution_context" in preflight_source


def synthetic_candidates(protocol: dict, contract: dict, checkpoint_path: Path) -> list[dict]:
    matrix = protocol["execution_contract"]["command_templates"]["train"][
        "matrix_contexts"
    ]
    capacities = list(dict.fromkeys(row["capacity_label"] for row in matrix))
    rows = []
    for capacity in capacities:
        for agent in contract["learned_agent_order"]:
            for seed in protocol["seed_plan"]["seeds"]:
                rows.append(
                    {
                        "agent_name": agent,
                        "seed": seed,
                        "capacity_label": capacity,
                        "update_index": 4,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "full_service_ready_byte_hit_rate": 0.5,
                        "workflow_continuity_rate": 0.5,
                        "transfer_mb_per_request": 1.0,
                        "end_to_end_workflow_delay": 1.0,
                        "runtime_contract_sha256": "r" * 64,
                        "resolved_agent_config": {},
                        "checkpoint_schedule": {},
                        "agent_scientific_config_semantic_sha256": "s" * 64,
                        "formal_training_execution_binding_sha256": "b" * 64,
                        "formal_protocol_semantic_sha256": protocol["hashes"][
                            "semantic_sha256"
                        ],
                        "execution_commit": "a" * 40,
                        "resolved_execution_context_sha256": "c" * 64,
                        "formal_agent_order_contract_semantic_sha256": contract[
                            "semantic_sha256"
                        ],
                        "typed_runtime_provenance": {
                            "execution_git_commit": "a" * 40,
                            "typed_catalog_fingerprint": "t" * 64,
                            "train_window_plan_identity": {},
                        },
                    }
                )
    return rows


def test_checkpoint_selection_and_freeze_preserve_authoritative_order(
    protocol: dict, contract: dict, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"nonformal-order-checkpoint")
    candidates = synthetic_candidates(protocol, contract, checkpoint)
    random.Random(13).shuffle(candidates)
    (tmp_path / "checkpoint_candidates.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    selection = dev_select(tmp_path, protocol)
    (tmp_path / "dev_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    assert selection["selected_agent_order"] == contract["learned_agent_order"]
    frozen = checkpoint_freeze(tmp_path, protocol)
    assert frozen["frozen_agent_order"] == contract["learned_agent_order"]
    drift = deepcopy(selection)
    drift["selected"][0], drift["selected"][5] = (
        drift["selected"][5],
        drift["selected"][0],
    )
    (tmp_path / "dev_selection.json").write_text(json.dumps(drift), encoding="utf-8")
    with pytest.raises(ValueError, match="order drift"):
        checkpoint_freeze(tmp_path, protocol)


def write_statistics_rows(path: Path, contract: dict, *, omit: str = "") -> None:
    rows = []
    for index, agent in enumerate(contract["main_benchmark_agent_order"]):
        if agent == omit:
            continue
        rows.append(
            {
                "seed": "7",
                "window_id": "w0",
                "workflow_id": "wf0",
                "agent_name": agent,
                "total_reward": str(100 - index),
            }
        )
    random.Random(path.name).shuffle(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_statistics(rows_path: Path, output_root: Path, contract: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_top_journal_statistics.py"),
            "--rows_path",
            str(rows_path),
            "--candidate_agent",
            contract["pairwise_statistics_identity"]["candidate_agent"],
            "--baseline_agents",
            *contract["pairwise_statistics_identity"]["baseline_agent_order"],
            "--metrics",
            "total_reward",
            "--bootstrap_samples",
            "20",
            "--output_root",
            str(output_root),
            "--formal-agent-order-contract-path",
            str(ORDER_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_statistics_mapping_reorder_is_invariant_and_wrong_pairing_rejected(
    contract: dict, tmp_path: Path
) -> None:
    rows_a = tmp_path / "a.csv"
    rows_b = tmp_path / "b.csv"
    write_statistics_rows(rows_a, contract)
    write_statistics_rows(rows_b, contract)
    first = run_statistics(rows_a, tmp_path / "out_a", contract)
    second = run_statistics(rows_b, tmp_path / "out_b", contract)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    payload_a = load(tmp_path / "out_a/paired_statistics.json")
    payload_b = load(tmp_path / "out_b/paired_statistics.json")
    assert payload_a["rows"] == payload_b["rows"]
    missing = tmp_path / "missing.csv"
    write_statistics_rows(missing, contract, omit="ppo")
    failed = run_statistics(missing, tmp_path / "out_missing", contract)
    assert failed.returncode != 0
    assert "agent matrix drift" in failed.stderr


def test_g14c_v7_all_references_and_protocol_binding_drift_are_rejected(
    protocol: dict
) -> None:
    invalid = ROOT / "artifacts/experiments/typed_model_cache_formal" / V7_RUN_ID
    with pytest.raises(FormalAgentOrderError, match="permanently invalid"):
        reject_permanently_invalid_run_references([invalid / "checkpoints/x.pt"])
    with pytest.raises(FormalExecutionError, match="permanently rejected"):
        reject_invalid_run_root(protocol, invalid)
    drift = deepcopy(protocol)
    drift["formal_agent_order_contract"]["semantic_sha256"] = "0" * 64
    drift = attach_hashes(drift)
    with pytest.raises(FormalExecutionError, match="order contract"):
        validate_protocol_v1_1(drift)


def test_holdout_capability_remains_false_and_readiness_v9_is_exact(
    protocol: dict,
) -> None:
    assert protocol["holdout_execution_contract"]["sealed"] is True
    assert protocol["holdout_execution_contract"]["opened"] is False
    assert protocol["holdout_execution_contract"]["consumed_permanently"] is False
    checks = {
        "g14c_v7_failure_registered",
        "formal_agent_order_contract_frozen",
        "producer_consumer_matrix_complete",
        "scientific_config_hash_unchanged",
        "protocol_fairness_command_order_reconciled",
        "training_commands_150_order_audited",
        "all_dev_commands_order_audited",
        "formal_support_scalability_order_audited",
        "full_15_agent_nonformal_rehearsal",
        "checkpoint_selection_freeze_order_stable",
        "statistics_order_invariant",
        "negative_validation_complete",
        "binding_context_order_hash_bound",
        "outer_nested_expansion_equal",
        "clean_worktree_without_local_venv",
        "clean_import_origin",
        "window_reachability_60_of_60",
        "real_preflight_completed",
        "real_tests_phase_completed",
        "phase_cell_resume_finalize_regression",
        "full_pytest_and_smoke_pass",
        "holdout_sealed",
        "no_formal_training_checkpoint_or_performance",
    }
    assert readiness_v9({name: True for name in checks}) == READY_V9_VERDICT
    blocked = {name: True for name in checks}
    blocked["statistics_order_invariant"] = False
    assert readiness_v9(blocked) == "BLOCKED_G14R7_READINESS_V9"
