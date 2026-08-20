from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluators.formal_window_consumption import (
    FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION,
    FormalWindowConsumptionError,
    load_contract,
    load_window_bundle_from_contract,
    load_window_plan,
    validate_contract,
    validate_window_plan_binding,
)
from src.evaluators.typed_model_cache_formal_execution import (
    FAILURE_CLASSIFICATIONS,
    READY_V4_VERDICT,
    FormalExecutionError,
    classify_phase_failure,
    readiness_v4,
    validate_protocol_v1_1,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820"
)
ARTIFACT_ROOT = (
    ROOT
    / "artifacts/analysis/typed_model_cache_formal_window_repair_20260820_g14r2_v1"
)
CONTRACT_PATH = CONFIG_ROOT / "formal_window_consumption_contract.json"
PROTOCOL_PATH = CONFIG_ROOT / "protocol_v1_2_manifest.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


CONTRACT = _json(CONTRACT_PATH)
PROTOCOL = _json(PROTOCOL_PATH)
REACHABILITY_ROWS = _json(ARTIFACT_ROOT / "window_reachability_rows.json")["rows"]


def _local_source_contract(tmp_path: Path) -> dict:
    contract = deepcopy(CONTRACT)
    source = tmp_path / "ngsim.csv"
    source.write_bytes(b"fixture")
    contract["source"]["path"] = str(source)
    contract["source"]["size_bytes"] = source.stat().st_size
    return contract


def _binding(
    contract: dict,
    *,
    split: str,
    plan_path: Path | None = None,
    max_rows: int = 11_850_526,
    mode: str = "formal",
    selector: str = "ordered",
    length: int = 24,
    rsu_layout: str = "auto_dominant_tight",
    vehicle_selection: str = "handoff_pressure",
) -> dict:
    return validate_window_plan_binding(
        contract=contract,
        plan_path=plan_path or Path(contract["window_plans"][split]["path"]),
        split=split,
        max_mobility_rows=max_rows,
        mobility_csv_path=contract["source"]["path"],
        window_selector=selector,
        window_length=length,
        rsu_layout=rsu_layout,
        primary_vehicle_selection=vehicle_selection,
        mode=mode,
    )


def test_contract_is_canonical_and_has_all_60_windows() -> None:
    report = load_contract(CONTRACT_PATH)
    assert report["formal_window_consumption_contract_version"] == (
        FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION
    )
    assert validate_contract(report)["window_count"] == 60


def test_contract_uses_explicit_full_safe_source_prefix() -> None:
    resolved = CONTRACT["resolved_source_range"]
    assert resolved == {
        "start_row_inclusive": 0,
        "end_row_exclusive": 11_850_526,
        "source_row_count": 11_850_526,
        "margin_rows": 0,
        "exceeds_source_behavior": "reject",
        "derivation": resolved["derivation"],
    }
    assert set(CONTRACT["split_required_source_rows"].values()) == {11_850_526}


@pytest.mark.parametrize(
    ("split", "expected"),
    [("train", 24), ("dev", 12), ("formal", 12), ("sealed_holdout", 12)],
)
def test_frozen_split_counts_are_unchanged(split: str, expected: int) -> None:
    assert sum(unit["split_name"] == split for unit in CONTRACT["evaluation_units"]) == expected
    assert CONTRACT["window_plans"][split]["window_count"] == expected


def test_split_semantic_hash_is_unchanged() -> None:
    expected = "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a"
    assert CONTRACT["split_semantic_sha256"] == expected
    assert PROTOCOL["identity"]["split_semantic_sha256"] == expected


def test_formal_binding_accepts_exact_train_plan(tmp_path: Path) -> None:
    report = _binding(_local_source_contract(tmp_path), split="train")
    assert report["status"] == "pass"
    assert report["complete_split_bound"] is True
    assert report["window_count"] == 24


def test_default_1500_row_scope_is_rejected_before_episode(tmp_path: Path) -> None:
    with pytest.raises(FormalWindowConsumptionError, match="max_mobility_rows"):
        _binding(_local_source_contract(tmp_path), split="train", max_rows=1500)


def test_unknown_frame_offset_identity_is_unreachable_without_loading_data() -> None:
    with pytest.raises(FormalWindowConsumptionError, match="not bound"):
        load_window_bundle_from_contract(
            contract_path=CONTRACT_PATH,
            split="train",
            window_id="unknown-provider-frame-offset",
            rsu_layout="auto_dominant_tight",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selector", "max_handoff_candidate", "ordered exact-offset"),
        ("length", 12, "length override"),
        ("rsu_layout", "auto", "RSU layout override"),
        ("vehicle_selection", "reward_rank", "vehicle selection identity"),
    ],
)
def test_semantic_cli_window_overrides_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    kwargs = {field: value}
    with pytest.raises(FormalWindowConsumptionError, match=message):
        _binding(_local_source_contract(tmp_path), split="train", **kwargs)


def test_wrong_source_path_is_rejected(tmp_path: Path) -> None:
    contract = _local_source_contract(tmp_path)
    other = tmp_path / "other.csv"
    other.write_bytes(b"fixture")
    with pytest.raises(FormalWindowConsumptionError, match="source path mismatch"):
        validate_window_plan_binding(
            contract=contract,
            plan_path=contract["window_plans"]["train"]["path"],
            split="train",
            max_mobility_rows=11_850_526,
            mobility_csv_path=other,
            window_selector="ordered",
            window_length=24,
            rsu_layout="auto_dominant_tight",
            primary_vehicle_selection="handoff_pressure",
        )


def test_source_size_mismatch_is_rejected(tmp_path: Path) -> None:
    contract = _local_source_contract(tmp_path)
    contract["source"]["size_bytes"] += 1
    with pytest.raises(FormalWindowConsumptionError, match="source size mismatch"):
        _binding(contract, split="train")


@pytest.mark.parametrize(
    ("field", "delta"),
    [
        ("frame_offset", 1),
        ("raw_frame_start", 1),
        ("raw_time_start", 100),
        ("source_segment_id", "_mismatch"),
    ],
)
def test_plan_identity_mismatch_is_rejected(
    tmp_path: Path, field: str, delta: int | str
) -> None:
    contract = _local_source_contract(tmp_path)
    plan = _json(Path(contract["window_plans"]["train"]["path"]))
    value = plan["selected_window_plan"][0][field]
    plan["selected_window_plan"][0][field] = value + delta
    path = tmp_path / f"mutated_{field}.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FormalWindowConsumptionError, match="identity mismatch"):
        _binding(contract, split="train", plan_path=path)


def test_result_field_in_window_plan_is_rejected(tmp_path: Path) -> None:
    plan = _json(Path(CONTRACT["window_plans"]["train"]["path"]))
    plan["selected_window_plan"][0]["reward"] = 1.0
    path = tmp_path / "leaky_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FormalWindowConsumptionError, match="performance field leaked"):
        load_window_plan(path)


def test_holdout_allows_identity_only_binding(tmp_path: Path) -> None:
    report = _binding(
        _local_source_contract(tmp_path),
        split="sealed_holdout",
        mode="identity_only",
    )
    assert report["status"] == "pass"
    assert report["window_count"] == 12


def test_holdout_rejects_formal_or_rehearsal_execution(tmp_path: Path) -> None:
    with pytest.raises(FormalWindowConsumptionError, match="identity-only"):
        _binding(_local_source_contract(tmp_path), split="sealed_holdout", mode="formal")


def test_identity_only_mode_is_reserved_for_holdout(tmp_path: Path) -> None:
    with pytest.raises(FormalWindowConsumptionError, match="reserved"):
        _binding(_local_source_contract(tmp_path), split="formal", mode="identity_only")


@pytest.mark.parametrize(
    "row",
    REACHABILITY_ROWS,
    ids=[f"{row['split']}:{row['window_id']}" for row in REACHABILITY_ROWS],
)
def test_each_frozen_window_has_exact_frame_time_provider_and_fingerprint_identity(
    row: dict,
) -> None:
    assert row["reachable"] is True
    assert row["errors"] == []
    assert len(range(row["observed_frame_interval"][0], row["observed_frame_interval"][1] + 1)) == 24
    assert row["observed_time_interval"][1] - row["observed_time_interval"][0] == 2300
    assert row["observed_provider_interval"][1] - row["observed_provider_interval"][0] == 23
    assert row["fingerprint_match"] is True
    assert row["observed_fingerprint"] == row["expected_fingerprint"]
    assert row["vehicle_coverage"]["minimum"] > 0
    assert row["metadata_only"] is (row["split"] == "sealed_holdout")


def test_reachability_summary_is_60_of_60_and_outcome_blind() -> None:
    report = _json(ARTIFACT_ROOT / "window_reachability_summary.json")
    assert report["status"] == "pass"
    assert report["window_count"] == report["reachable_count"] == 60
    assert report["split_reachable_counts"] == {
        "train": 24,
        "dev": 12,
        "formal": 12,
        "sealed_holdout": 12,
    }
    assert report["provider_frame_count"] == 73_871
    assert report["provider_frame_count_match"] is True
    assert report["provider_identity_recomputed_from_raw_source"] is True
    assert report["run_local_identity_recomputed_from_raw_source"] is True
    assert report["holdout_metadata_only"] is True
    assert report["agent_or_policy_executed"] is False
    assert report["performance_fields_read"] is False


def test_boundary_rehearsal_covers_minimum_and_maximum_offset_per_split() -> None:
    report = _json(ARTIFACT_ROOT / "boundary_window_rehearsal.json")
    assert report["status"] == "pass"
    assert len(report["rows"]) == 8
    for split in ("train", "dev", "formal", "sealed_holdout"):
        rows = [row for row in report["rows"] if row["split"] == split]
        assert len(rows) == 2
        assert all(row["reachable"] and row["fingerprint_match"] for row in rows)
        assert all(row["metadata_only"] is (split == "sealed_holdout") for row in rows)


def test_training_and_benchmark_fingerprints_have_exact_parity() -> None:
    report = _json(ARTIFACT_ROOT / "training_benchmark_fingerprint_parity.json")
    assert report["status"] == "pass"
    assert report["window_count"] == 60
    assert report["same_loader_identity"] is True
    assert report["same_preprocessing_identity"] is True
    assert report["all_fingerprints_match"] is True
    assert all(row["training_fingerprint"] == row["benchmark_fingerprint"] for row in report["rows"])


def test_all_150_training_commands_have_explicit_range_and_unique_outputs() -> None:
    report = _json(ARTIFACT_ROOT / "training_command_validation_150.json")
    assert report["status"] == "pass"
    assert report["training_command_count"] == 150
    assert report["unique_output_count"] == 150
    assert all(row["resolved_source_rows"] == 11_850_526 for row in report["rows"])
    assert all(row["checkpoint_cadence"] == 4 for row in report["rows"])
    assert not any(row["holdout"] for row in report["rows"])


def test_all_formal_and_support_commands_have_resolved_window_binding() -> None:
    report = _json(ARTIFACT_ROOT / "formal_command_validation.json")
    assert report["status"] == "pass"
    assert report["formal_command_count"] == 30
    assert all(row["resolved_source_rows"] == 11_850_526 for row in report["rows"])
    assert all(row["window_binding"] in {"dev", "formal"} for row in report["rows"])
    assert not any(row["holdout"] for row in report["rows"])


def test_command_templates_reject_implicit_range_or_window_override() -> None:
    templates = _json(ARTIFACT_ROOT / "formal_command_templates_v1_2.json")["templates"]
    bound = [item for item in templates.values() if "--max_mobility_rows" in item.get("argv", [])]
    assert bound
    for template in bound:
        argv = template["argv"]
        assert argv[argv.index("--max_mobility_rows") + 1] == "11850526"
        assert argv[argv.index("--window_selector") + 1] == "ordered"
        assert argv[argv.index("--window_length") + 1] == "24"
        assert argv.count("--max_mobility_rows") == 1
        assert argv.count("--window_selector") == 1


def test_protocol_v1_2_validates_and_v1_1_is_explicitly_invalidated() -> None:
    assert validate_protocol_v1_1(PROTOCOL)["status"] == "pass"
    assert PROTOCOL["typed_model_cache_formal_protocol_version"] == "1.2.0"
    assert PROTOCOL["supersession"]["old_protocol_status"] == (
        "invalid_before_performance_execution"
    )
    assert PROTOCOL["supersession"]["formal_performance_observed"] is False
    assert PROTOCOL["hashes"]["semantic_sha256"] == (
        "718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9"
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_protocol_v1_2_rejects_non_finite_values(bad: float) -> None:
    mutated = deepcopy(PROTOCOL)
    mutated["training_budget"]["batch_size"] = bad
    with pytest.raises(FormalExecutionError, match="non-finite"):
        validate_protocol_v1_1(mutated)


def test_protocol_restart_diff_changes_only_execution_contract_families() -> None:
    report = _json(ARTIFACT_ROOT / "protocol_restart_diff.json")
    assert report["status"] == "pass"
    assert report["protocol_semantic_hash_changed"] is True
    assert report["split_semantic_sha256"] == CONTRACT["split_semantic_sha256"]
    assert all(report["scientific_fields_unchanged"].values())


def test_phase_ledger_schema_and_failure_enum_are_complete() -> None:
    schema = _json(ARTIFACT_ROOT / "phase_ledger_schema.json")
    validation = _json(ARTIFACT_ROOT / "phase_ledger_validation.json")
    assert schema["schema_version"] == "2.0.0"
    assert set(schema["failure_classifications"]) == set(FAILURE_CLASSIFICATIONS)
    assert validation["status"] == "pass"
    assert validation["success_chain"]["record_count"] == 2
    assert validation["failure_chain"]["record_count"] == 2
    assert validation["running_record_supported"] is True
    assert validation["terminal_immutable"] is True
    assert validation["JSON_round_trip"] is True
    assert validation["deterministic_hash"] is True


@pytest.mark.parametrize(
    ("phase", "return_code", "message", "expected"),
    [
        ("train", 75, "temporary runner loss", "infrastructure_retryable"),
        ("train", 1, "frame_offset exceeds provider", "data_window_unreachable"),
        ("tests", 1, "assertion failed", "test_failure"),
        ("train", 1, "optimizer failed", "training_failure"),
        ("formal_gate", 1, "checksum differs", "artifact_integrity_failure"),
        ("preflight", 130, "interrupted", "user_interruption"),
        ("preflight", 64, "host failure", "infrastructure_terminal"),
        ("preflight", None, "executor exception", "implementation_error"),
        ("preflight", 1, "protocol mismatch", "protocol_mismatch"),
    ],
)
def test_failure_classification_is_total_and_retry_is_75_only(
    phase: str, return_code: int | None, message: str, expected: str
) -> None:
    assert classify_phase_failure(
        phase=phase, return_code=return_code, message=message
    ) == expected


def test_g14c_v2_return_code_one_is_not_retryable() -> None:
    report = _json(ARTIFACT_ROOT / "failure_classification_validation.json")
    assert report["g14c_v2_return_code_1_classification"] == "data_window_unreachable"
    assert report["g14c_v2_retry_eligible"] is False
    assert report["return_code_75_only"] == "infrastructure_retryable"


def test_tiny_rehearsal_covers_four_agents_two_seeds_two_capacities() -> None:
    report = _json(ARTIFACT_ROOT / "tiny_training_rehearsal.json")
    assert report["status"] == "pass"
    assert report["agents"] == ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"]
    assert report["seeds"] == [7, 13]
    assert report["capacities_mb"] == [288.0, 576.0]
    assert report["training_cell_count"] == 16
    assert all(cell["status"] == "pass" for cell in report["training_cells"])
    assert all(cell["checkpoint_restore_and_provenance"] == "compatible" for cell in report["training_cells"])


def test_tiny_rehearsal_is_nonformal_and_keeps_holdout_closed() -> None:
    report = _json(ARTIFACT_ROOT / "tiny_training_rehearsal.json")
    assert report["formal_checkpoint_count"] == 0
    assert report["formal_episode_count"] == 0
    assert report["formal_performance_result_count"] == 0
    assert report["holdout_opened"] is False
    assert report["holdout_episode_count"] == 0
    assert report["performance_claims"] == []


def test_sa_config_checkpoint_cadence_and_support_binding_are_frozen() -> None:
    companion = _json(CONFIG_ROOT / "agent_training_configs.json")
    assert companion["agents"]["sa_ghmappo"]["auxiliary_coef"] == 0.06
    assert PROTOCOL["training_budget"]["checkpoint_frequency_updates"] == 4
    support = PROTOCOL["ablation_and_support"]["support_setting_matrix"]["settings"]
    assert support
    assert all(item["levels"] for item in support)


def test_holdout_seal_revalidation_is_identity_only() -> None:
    report = _json(ARTIFACT_ROOT / "holdout_seal_revalidation.json")
    assert report == {
        "status": "pass",
        "sealed": True,
        "opened": False,
        "consumed_permanently": False,
        "identity_reachability_count": 12,
        "metadata_only": True,
        "agent_or_episode_execution": False,
        "performance_fields_read": False,
    }


def test_readiness_v4_success_and_missing_check_failure() -> None:
    review = _json(ARTIFACT_ROOT / "readiness_review_v4.json")
    assert readiness_v4(review["checks"]) == READY_V4_VERDICT
    assert review["verdict"] == READY_V4_VERDICT
    missing = dict(review["checks"])
    missing.pop("window_reachability_60_of_60")
    with pytest.raises(FormalExecutionError, match="check set mismatch"):
        readiness_v4(missing)


def test_readiness_v4_blocks_when_any_required_check_fails() -> None:
    checks = dict(_json(ARTIFACT_ROOT / "readiness_review_v4.json")["checks"])
    checks["ledger_append_chain"] = False
    assert readiness_v4(checks) == "BLOCKED_G14R2_READINESS_V4"
