from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agents.registry import build_agent
from src.evaluators.main_results_support import aggregate_rows
from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    CommandResult,
    FormalExecutionError,
    PHASE_ORDER,
    PRIMARY_ENDPOINTS,
    READY_VERDICT,
    build_scalability_setting_matrix,
    build_support_setting_matrix,
    expand_command_template,
    protocol_hash_changes_on_mutation,
    readiness_v3,
    reconcile_primary_endpoint_row,
    stable_setting_identity,
    support_setting_by_id,
    validate_command_templates,
    validate_no_holdout_capability,
    validate_phase_ledger,
    validate_protocol_v1_1,
)
from src.metrics.cache_efficiency_metrics import (
    cache_efficiency_row_fields,
    reduce_cache_efficiency_events,
)
from src.runtime.formal_training_contract import (
    FormalTrainingContractError,
    audited_agent_config,
    checkpoint_schedule_metadata,
    checkpoint_snapshot_indices,
    resolve_training_contract,
    should_save_checkpoint,
    validate_resume_checkpoint_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/protocol_v1_1_manifest.json"
)
AGENT_CONFIG_PATH = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/agent_training_configs.json"
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agent_companion() -> dict:
    return json.loads(AGENT_CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve(
    agent: str,
    *,
    protocol: dict | None = None,
    companion: dict | None = None,
    cli: dict | None = None,
):
    return resolve_training_contract(
        agent_name=agent,
        profile_defaults={"episodes": 2, "update_every": 1, "batch_size": 8, "max_steps": 6},
        cli_values=cli or {},
        formal_protocol=protocol,
        agent_config_companion=companion,
    )


def _typed_event(event_id: str, *, full: bool, base: float | None = 160.0, adapter: float | None = 40.0, transfers: dict | None = None) -> dict:
    rows = [
        {"object_id": "base:b", "object_type": "base_model", "resident_size_mb": base, "transfer_size_mb": 160.0},
        {"object_id": "adapter:a", "object_type": "adapter", "adapter_id": "a", "resident_size_mb": adapter, "transfer_size_mb": 40.0},
    ]
    lookups = [
        {"object_id": row["object_id"], "resident": full, "object_type": row["object_type"]}
        for row in rows
    ]
    return {
        "event_id": event_id,
        "event_schema_version": "1.3.0",
        "event_type": "request",
        "time_index": 1,
        "episode_step_index": 1,
        "vehicle_id": "v",
        "workflow_id": "w",
        "node_id": "n",
        "object_id": "adapter:a",
        "adapter_id": "a",
        "object_type": "adapter",
        "size_mb": 40.0,
        "request_rsu_id": "r",
        "selected_target_rsu_id": "r" if full else None,
        "served_rsu_id": "r" if full else None,
        "predicted_next_rsu_id": None,
        "predicted_handoff_target_rsu_id": None,
        "hit_source": "current_rsu" if full else "cloud",
        "cache_lookup_performed": full,
        "cache_hit": full,
        "was_cached_before": full,
        "admission_requested": False,
        "admission_added": False,
        "admission_reason": "not_requested",
        "cache_target_rsu_id": None,
        "eviction_occurred": False,
        "eviction_policy": "lru",
        "evicted_object_id": None,
        "evicted_adapter_id": None,
        "eviction_reason": "not_occurred",
        "adapter_transfer_size_mb": float((transfers or {}).get("adapter", 0.0)),
        "state_migration_size_mb": float((transfers or {}).get("workflow_state", 0.0)),
        "transfer_source": "typed_catalog",
        "migration_requested": bool((transfers or {}).get("workflow_state")),
        "migration_realized": bool((transfers or {}).get("workflow_state")),
        "cache_capacity_enabled": False,
        "cache_capacity_unit": "mb",
        "cache_capacity_before": None,
        "cache_used_before": None,
        "cache_remaining_before": None,
        "cache_capacity_after": None,
        "cache_used_after": None,
        "cache_remaining_after": None,
        "action_id": 3,
        "action_name": "steady",
        "cache_strategy": "none",
        "offload_mode": "rsu" if full else "cloud",
        "service_success": True,
        "stall_occurred": False,
        "handoff_event_count": 0,
        "eviction_count": 0,
        "evicted_object_ids": [],
        "evicted_adapter_ids": [],
        "evicted_size_mb_sum": 0.0,
        "requested_object_size_mb": 40.0,
        "capacity_rejection_reason": None,
        "admitted_object_id": None,
        "admitted_adapter_id": None,
        "admitted_size_mb": None,
        "evicted_sizes_mb": [],
        "typed_model_cache_contract_version": "1.0.0",
        "model_cache_profile_id": "typed_base_adapter_state_v1",
        "requested_typed_objects": rows,
        "dependency_bundle": {"ordered_object_ids": ["base:b", "adapter:a"]},
        "per_object_lookup_results": lookups,
        "base_model_hit": full,
        "adapter_hit": full,
        "joint_model_hit": full,
        "workflow_state_ready": True,
        "full_service_ready": full,
        "missing_object_types": [] if full else ["base_model", "adapter"],
        "incompatibility_reason": None,
        "compatibility_result": "compatible",
        "admitted_typed_objects": [],
        "evicted_typed_objects": [],
        "admitted_mb_by_type": {},
        "evicted_mb_by_type": {},
        "transfer_mb_by_type": dict(transfers or {}),
        "typed_capacity_snapshot": None,
        "atomic_transaction_status": "not_requested",
        "orphan_count": 0,
    }


def _summary(events: list[dict]) -> dict:
    return {
        "cache_event_schema_version": "1.3.0",
        "cache_event_trace": events,
        "system_metrics": {
            "workflow_continuity_rate": 0.75,
            "end_to_end_workflow_delay": 12.0,
        },
    }


# Checkpoint 1-5
def test_01_legacy_default_saves_each_update() -> None:
    resolved = _resolve("ppo")
    assert resolved.checkpoint_every_updates == 1
    assert checkpoint_snapshot_indices(4, resolved.checkpoint_every_updates) == [1, 2, 3, 4]


def test_02_formal_every_four_updates(protocol: dict, agent_companion: dict) -> None:
    resolved = _resolve("ppo", protocol=protocol, companion=agent_companion)
    assert resolved.checkpoint_every_updates == 4
    assert checkpoint_snapshot_indices(32, 4) == [4, 8, 12, 16, 20, 24, 28, 32]


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_03_invalid_frequency(value) -> None:
    with pytest.raises(FormalTrainingContractError):
        _resolve("ppo", cli={"checkpoint_every_updates": value})


def test_04_resume_frequency_consistent() -> None:
    metadata = {"checkpoint_schedule": checkpoint_schedule_metadata(checkpoint_every_updates=4, expected_update_count=32)}
    assert validate_resume_checkpoint_schedule(metadata, checkpoint_every_updates=4) == 4
    with pytest.raises(FormalTrainingContractError, match="mismatch"):
        validate_resume_checkpoint_schedule(metadata, checkpoint_every_updates=1)


def test_05_actual_checkpoint_indices() -> None:
    actual = [index for index in range(1, 33) if should_save_checkpoint(index, 4)]
    assert actual == [4, 8, 12, 16, 20, 24, 28, 32]


# SA config 6-10
def test_06_sa_auxiliary_006_passed(protocol: dict, agent_companion: dict) -> None:
    resolved = _resolve("sa_ghmappo", protocol=protocol, companion=agent_companion)
    assert resolved.agent_config["auxiliary_coef"] == 0.06
    agent = build_agent("sa_ghmappo", random_seed=7, batch_size=1, **resolved.agent_config)
    assert audited_agent_config(agent, resolved.agent_config)["auxiliary_coef"] == 0.06


def test_07_resolved_agent_config(protocol: dict, agent_companion: dict) -> None:
    resolved = _resolve("sa_ghmappo", protocol=protocol, companion=agent_companion)
    assert resolved.to_dict()["agent_config"]["auxiliary_coef"] == 0.06


def test_08_checkpoint_metadata_records_agent_config(protocol: dict, agent_companion: dict) -> None:
    resolved = _resolve("sa_ghmappo", protocol=protocol, companion=agent_companion)
    metadata = {"resolved_agent_config": resolved.agent_config}
    assert metadata["resolved_agent_config"]["auxiliary_coef"] == 0.06


def test_09_agent_config_mismatch_rejected(protocol: dict, agent_companion: dict) -> None:
    bad = deepcopy(agent_companion)
    bad["agents"]["sa_ghmappo"]["auxiliary_coef"] = 0.1
    with pytest.raises(FormalTrainingContractError, match="mismatch"):
        _resolve("sa_ghmappo", protocol=protocol, companion=bad)


def test_10_other_agent_has_no_sa_field(protocol: dict, agent_companion: dict) -> None:
    assert "auxiliary_coef" not in _resolve("ppo", protocol=protocol, companion=agent_companion).agent_config


# Endpoints 11-21
def test_11_full_service_ready_byte_hit_hand_case() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True), _typed_event("b", full=False)], schema_version="1.3.0")
    assert result.type_aware_metrics["full_service_ready_byte_hit_rate"] == 0.5


def test_12_shared_base_denominator_once_per_request() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True)], schema_version="1.3.0")
    assert result.type_aware_metrics["requested_service_dependency_mb"] == 200.0


def test_13_partial_readiness_zeroes_whole_request_numerator() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=False)], schema_version="1.3.0")
    assert result.type_aware_metrics["full_service_ready_dependency_mb"] == 0.0
    assert result.type_aware_metrics["full_service_ready_byte_hit_rate"] == 0.0


def test_14_missing_dependency_bytes_are_partial() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True, base=None)], schema_version="1.3.0")
    assert result.type_aware_metrics["full_service_ready_byte_hit_rate"] is None
    assert result.type_aware_metrics["requested_dependency_byte_coverage_rate"] == 0.0


def test_15_zero_denominator_is_null() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True, base=0.0, adapter=0.0)], schema_version="1.3.0")
    assert result.type_aware_metrics["full_service_ready_byte_hit_rate"] is None


def test_16_transfer_components() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True, transfers={"base_model": 160.0, "adapter": 40.0})], schema_version="1.3.0")
    typed = result.type_aware_metrics
    assert typed["base_model_transfer_mb"] == 160.0
    assert typed["adapter_transfer_mb"] == 40.0


def test_17_transfer_per_request() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True, transfers={"adapter": 40.0}), _typed_event("b", full=False)], schema_version="1.3.0")
    assert result.type_aware_metrics["transfer_mb_per_request"] == 20.0


def test_18_state_migration_in_primary_transfer() -> None:
    result = reduce_cache_efficiency_events([_typed_event("a", full=True, transfers={"workflow_state": 20.0})], schema_version="1.3.0")
    assert result.type_aware_metrics["workflow_state_migration_transfer_mb"] == 20.0
    assert result.type_aware_metrics["transfer_mb_per_request"] == 20.0


def test_19_legacy_unavailable() -> None:
    result = reduce_cache_efficiency_events([], schema_version="1.2.0")
    assert result.type_aware_metrics["availability"] == "unavailable"


def test_20_row_reducer_reconciliation() -> None:
    summary = _summary([_typed_event("a", full=True)])
    row = {**cache_efficiency_row_fields(summary), **summary["system_metrics"]}
    assert reconcile_primary_endpoint_row(summary, row)["status"] == "pass"


def test_21_nullable_aggregate() -> None:
    aggregate = aggregate_rows([{"agent": "a", "full_service_ready_byte_hit_rate": None}], ["agent"], ["full_service_ready_byte_hit_rate"])
    assert aggregate["a"]["metrics"]["full_service_ready_byte_hit_rate"]["mean"] is None


# Support 22-29
def test_22_concrete_or_explicit_unavailable_values_exist() -> None:
    matrix = build_support_setting_matrix()
    assert all(item["values"] and item["status"] for item in matrix["settings"])


def test_23_stable_setting_identity() -> None:
    assert stable_setting_identity("x", {"a": 1}) == stable_setting_identity("x", {"a": 1})


def test_24_typed_runtime_is_frozen(protocol: dict) -> None:
    assert protocol["identity"]["typed_runtime_contract_hashes_by_capacity"]


def test_25_fairness_assets_are_persisted() -> None:
    assert (PROTOCOL_PATH.parent / "fairness_medium_576mb.json").is_file()


def test_26_checkpoint_provenance_is_required(protocol: dict) -> None:
    assert "checkpoint" in json.dumps(protocol["execution_contract"]["command_templates"])


def test_27_support_cli_override_has_no_free_value_flag(protocol: dict) -> None:
    template = protocol["execution_contract"]["command_templates"]["formal_support"]["argv"]
    assert "--setting-id" in template and "--prediction_noise_std" not in template


def test_28_unsupported_setting_is_explicitly_unavailable(protocol: dict) -> None:
    matrix = protocol["ablation_and_support"]["support_setting_matrix"]
    level = next(level for item in matrix["settings"] if item["parameter"] == "object_size_scale" for level in item["levels"])
    setting = support_setting_by_id(protocol, level["setting_id"])
    assert setting["status"] == "unavailable_pre_execution"


def test_29_support_output_provenance_fields(protocol: dict) -> None:
    assert protocol["execution_contract"]["holdout_capability"] is False
    assert build_scalability_setting_matrix()["settings"][0]["levels"][0]["setting_id"]


# Phase 30-40
def _phase_runner(protocol: dict, tmp_path: Path) -> AppendOnlyPhaseRunner:
    return AppendOnlyPhaseRunner(protocol=protocol, output_root=tmp_path / "run")


def _complete_phase(runner: AppendOnlyPhaseRunner, phase: str, input_hash: str | None = None, results: list[int] | None = None):
    output = f"{phase}.json"
    codes = list(results or [0])
    def execute(_command):
        code = codes.pop(0)
        if code == 0:
            (runner.output_root / output).write_text("{}", encoding="utf-8")
        return CommandResult(code)
    return runner.run_phase(phase, command=[] if phase == "complete_without_holdout" else ["ok"], input_hash=input_hash or phase, expected_outputs=[] if phase == "complete_without_holdout" else [output], executor=execute)


def test_30_phase_dry_command_expansion(protocol: dict) -> None:
    report = validate_command_templates(protocol["execution_contract"]["command_templates"], protocol["execution_contract"]["default_expansion_context"])
    assert report["status"] == "pass"


def test_31_phase_order(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    with pytest.raises(FormalExecutionError, match="phase order"):
        _complete_phase(runner, "tests")


def test_32_append_only(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    _complete_phase(runner, "preflight")
    records = [json.loads(line) for line in runner.ledger_path.read_text().splitlines()]
    assert [record["status"] for record in records] == ["running", "completed"]
    assert records[1]["previous_record_hash"] == records[0]["current_record_hash"]


def test_33_resume_skips_hash_match(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    _complete_phase(runner, "preflight", "same")
    resumed = AppendOnlyPhaseRunner(protocol=protocol, output_root=runner.output_root, resume=True)
    assert _complete_phase(resumed, "preflight", "same")["status"] == "skipped_completed_hash_match"


def test_34_hash_mismatch_fails(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    _complete_phase(runner, "preflight", "a")
    with pytest.raises(FormalExecutionError, match="input hash mismatch"):
        _complete_phase(runner, "preflight", "b")


def test_35_failed_phase_is_terminal(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    with pytest.raises(FormalExecutionError, match="phase failed"):
        _complete_phase(runner, "preflight", results=[1])
    with pytest.raises(FormalExecutionError, match="terminal"):
        _complete_phase(runner, "preflight")


def test_36_infrastructure_retry_once(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    event = _complete_phase(runner, "preflight", results=[75, 0])
    assert event["attempts"] == 2


def test_37_formal_sequence_reaches_complete_without_holdout(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    for phase in PHASE_ORDER:
        _complete_phase(runner, phase)
    assert runner.events()[-1]["phase"] == "complete_without_holdout"
    with pytest.raises(FormalExecutionError, match="training is forbidden"):
        _complete_phase(runner, "train")


def test_38_holdout_is_inaccessible() -> None:
    with pytest.raises(FormalExecutionError, match="no holdout"):
        validate_no_holdout_capability(["runner", "--holdout"])


def test_39_output_root_conflict(protocol: dict, tmp_path: Path) -> None:
    root = tmp_path / "conflict"
    root.mkdir()
    (root / "foreign").write_text("x")
    with pytest.raises(FormalExecutionError, match="conflict"):
        AppendOnlyPhaseRunner(protocol=protocol, output_root=root)


def test_40_complete_command_expansion() -> None:
    assert expand_command_template(["x", "{agent}"], {"agent": "ppo"}) == ["x", "ppo"]
    with pytest.raises(FormalExecutionError, match="unresolved"):
        expand_command_template(["{missing}"], {})


# Protocol 41-48
def test_41_v1_invalid_record(protocol: dict) -> None:
    assert protocol["supersession"]["old_protocol_status"] == "invalid_before_execution"


def test_42_v1_1_canonical_hash(protocol: dict) -> None:
    assert validate_protocol_v1_1(protocol)["status"] == "pass"


def test_43_split_hash_unchanged(protocol: dict) -> None:
    assert protocol["identity"]["split_semantic_sha256"] == "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a"


def test_44_semantic_mutation_changes_hash(protocol: dict) -> None:
    assert protocol_hash_changes_on_mutation(protocol, "training_budget.checkpoint_frequency_updates", 5)


def test_45_readiness_missing_consumer_rejected() -> None:
    with pytest.raises(FormalExecutionError, match="check set mismatch"):
        readiness_v3({"holdout_sealed": True})


def test_46_readiness_success() -> None:
    checks = {key: True for key in ["protocol_fields_have_runtime_consumers", "agent_commands_expand", "checkpoint_frequency_consistent", "sa_auxiliary_consistent", "primary_endpoint_producer_exists", "primary_endpoint_reconciliation", "support_values_concrete_or_unavailable", "typed_support_provenance", "phase_runner_dry_run", "fairness_manifests_persisted", "runtime_configs_persisted", "command_templates_persisted", "output_schema_exists", "clean_worktree_execution_plan", "holdout_sealed"]}
    assert readiness_v3(checks) == READY_VERDICT


def test_47_json_round_trip(protocol: dict) -> None:
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_48_nan_inf_rejected(protocol: dict, bad: float) -> None:
    mutated = deepcopy(protocol)
    mutated["training_budget"]["batch_size"] = bad
    with pytest.raises(FormalExecutionError, match="non-finite"):
        validate_protocol_v1_1(mutated)


# Ledger v2 49-55
def test_49_terminal_timing_and_wall_clock(protocol: dict, tmp_path: Path) -> None:
    clocks = iter(
        [
            datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 20, 0, 0, 1, tzinfo=timezone.utc),
        ]
    )
    monotonic = iter([10.0, 11.0])
    runner = AppendOnlyPhaseRunner(
        protocol=protocol,
        output_root=tmp_path / "timed",
        clock=lambda: next(clocks),
        monotonic_clock=lambda: next(monotonic),
    )
    event = _complete_phase(runner, "preflight")
    assert event["started_at"] <= event["completed_at"]
    assert event["wall_clock_seconds"] == 1.0
    assert validate_phase_ledger(runner.events())["terminal_phase_count"] == 1


def test_50_running_record_can_resume_by_appending(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    started = datetime.now(timezone.utc).isoformat()
    runner._append_record(  # noqa: SLF001 - contract-level crash/resume fixture.
        runner._base_record(  # noqa: SLF001
            phase="preflight",
            status="running",
            started_at=started,
            completed_at=None,
            wall_clock_seconds=None,
            input_hash="same",
            output_hash=None,
            commands=[["ok"]],
            return_code=None,
            retry_count=0,
            failure_classification=None,
            failure_message_reference=None,
        )
    )
    resumed = AppendOnlyPhaseRunner(
        protocol=protocol, output_root=runner.output_root, resume=True
    )
    event = _complete_phase(resumed, "preflight", "same")
    assert event["status"] == "completed"
    assert [row["status"] for row in resumed.events()] == [
        "running",
        "running",
        "completed",
    ]


def test_51_invalid_ledger_timestamp_is_rejected(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    with pytest.raises(FormalExecutionError, match="timestamp"):
        runner._append_record(  # noqa: SLF001
            runner._base_record(  # noqa: SLF001
                phase="preflight",
                status="running",
                started_at="not-a-timestamp",
                completed_at=None,
                wall_clock_seconds=None,
                input_hash="x",
                output_hash=None,
                commands=[],
                return_code=None,
                retry_count=0,
                failure_classification=None,
                failure_message_reference=None,
            )
        )


def test_52_missing_failure_classification_is_rejected(
    protocol: dict, tmp_path: Path
) -> None:
    runner = _phase_runner(protocol, tmp_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with pytest.raises(FormalExecutionError, match="failure classification"):
        runner._append_record(  # noqa: SLF001
            runner._base_record(  # noqa: SLF001
                phase="preflight",
                status="failed",
                started_at=timestamp,
                completed_at=timestamp,
                wall_clock_seconds=0.0,
                input_hash="x",
                output_hash=None,
                commands=[],
                return_code=1,
                retry_count=0,
                failure_classification=None,
                failure_message_reference="failure",
            )
        )


def test_53_retry_code_and_classification_must_match(
    protocol: dict, tmp_path: Path
) -> None:
    runner = _phase_runner(protocol, tmp_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with pytest.raises(FormalExecutionError, match="return code 75"):
        runner._append_record(  # noqa: SLF001
            runner._base_record(  # noqa: SLF001
                phase="preflight",
                status="failed",
                started_at=timestamp,
                completed_at=timestamp,
                wall_clock_seconds=0.0,
                input_hash="x",
                output_hash=None,
                commands=[],
                return_code=75,
                retry_count=1,
                failure_classification="implementation_error",
                failure_message_reference="failure",
            )
        )


def test_54_hash_chain_tamper_is_rejected(protocol: dict, tmp_path: Path) -> None:
    runner = _phase_runner(protocol, tmp_path)
    _complete_phase(runner, "preflight")
    records = runner.events()
    records[1]["previous_record_hash"] = "tampered"
    with pytest.raises(FormalExecutionError, match="previous hash mismatch"):
        validate_phase_ledger(records)


def test_55_ledger_json_round_trip_preserves_hashes(
    protocol: dict, tmp_path: Path
) -> None:
    runner = _phase_runner(protocol, tmp_path)
    _complete_phase(runner, "preflight")
    records = runner.events()
    round_tripped = json.loads(json.dumps(records, allow_nan=False))
    assert round_tripped == records
    assert validate_phase_ledger(round_tripped)["last_record_hash"] == records[-1][
        "current_record_hash"
    ]
