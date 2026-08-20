from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.evaluators.typed_model_cache_formal_protocol import (
    FormalProtocolError,
    HoldoutAccessError,
    InsufficientWindowError,
    append_holdout_execution_record,
    assert_no_semantic_cli_overrides,
    assert_result_blind_windows,
    attach_hashes,
    build_agent_matrix,
    build_capacity_strata,
    build_claim_evidence_template,
    build_formal_protocol,
    build_historical_registry,
    build_holdout_seal,
    build_split_manifest,
    build_statistics_protocol,
    canonical_json_bytes,
    canonical_sha256,
    extract_selected_window_plan,
    interval_relation,
    protocol_hash_changes_on_mutation,
    readiness_verdict,
    scan_ngsim_intervals,
    semantic_projection,
    validate_protocol_manifest,
    validate_split_access,
)
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


ROOT = Path(__file__).resolve().parents[1]


def make_window(index: int, *, run: str = "i_80_run_001", window_id: str | None = None) -> dict:
    start = 100 + index * 48
    time_start = 1_000_000 + start * 100
    return {
        "window_id": window_id or f"w{index}",
        "frame_offset": start,
        "provider_frame_offset": start,
        "window_length": 24,
        "time_index_start": time_start,
        "time_index_end": time_start + 2300,
        "raw_frame_start": start,
        "raw_frame_end": start + 23,
        "raw_time_start": time_start,
        "raw_time_end": time_start + 2300,
        "source_segment_id": "i_80",
        "source_segment_run_id": run,
        "source_location": "i_80",
        "segment_frame_start": start,
        "segment_frame_end": start + 23,
        "sampling_interval": 100,
        "active_vehicle_count_min": 2,
        "active_vehicle_count_mean": 3.0,
        "active_vehicle_count_max": 4,
        "window_class": "coverage_only_result_blind",
        "recommended_rsu_layout": "auto_dominant_tight",
    }


@pytest.fixture()
def candidate_inventory() -> dict:
    return attach_hashes(
        {
            "candidate_window_inventory_version": "1.0.0",
            "parameters": {
                "split_generation_seed": 1401,
                "tie_break": "sha256 then id",
                "window_length": 24,
                "minimum_vehicle_count": 2,
                "runner_prefix_max_mobility_rows": 5_000_000,
            },
            "candidates": [make_window(index) for index in range(80)],
        }
    )


@pytest.fixture()
def split_bundle(candidate_inventory: dict) -> tuple[dict, dict, dict]:
    return build_split_manifest(
        candidate_inventory,
        counts={"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12},
        minimum_gap_frames=24,
        created_at="2026-08-20T00:00:00+00:00",
    )


@pytest.fixture()
def runtime_contract() -> dict:
    raw = yaml.safe_load(
        (ROOT / "configs/benchmark/typed_model_cache_controlled_lru.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    return resolve_model_cache_runtime(raw, root=ROOT)


@pytest.fixture()
def protocol(split_bundle: tuple[dict, dict, dict], runtime_contract: dict) -> dict:
    split, _, _ = split_bundle
    registry = attach_hashes(
        {
            "historical_window_usage_registry_version": "1.0.0",
            "records": [],
        }
    )
    capacities = build_capacity_strata(runtime_contract)
    seal = build_holdout_seal(split)
    return build_formal_protocol(
        split_manifest=split,
        historical_registry=registry,
        runtime_contract=runtime_contract,
        runtime_hashes_by_capacity={
            item["stratum"]: f"runtime-{item['stratum']}" for item in capacities["strata"]
        },
        capacity_strata=capacities,
        dataset_hashes={"ngsim_sha256": "n", "alibaba_batch_task_sha256": "a", "typed_catalog_file_sha256": "c"},
        workflow_ids=["j_3", "j_8", "j_15"],
        fairness_manifest_version="1.1.0",
        holdout_seal=seal,
        created_at="2026-08-20T00:00:00+00:00",
    )


def synthetic_inventory_internal() -> dict:
    frames = []
    by_time = {}
    by_raw = {}
    for index in range(200):
        item = {
            "source_segment_id": "peachtree",
            "source_segment_run_id": "peachtree_run_001",
            "raw_frame": index,
            "global_time": 2_000_000_000 + index * 100,
        }
        frames.append(item)
        by_time[("peachtree", item["global_time"])] = item
        by_raw[("peachtree", index)] = [item]
    return {
        "frame_lookup_by_time": by_time,
        "frame_lookup_by_raw": by_raw,
        "prefix_frames": frames,
    }


def test_ngsim_prefix_cutoff_marks_only_a_frame_that_straddles_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ngsim.csv"
    rows = [
        "Vehicle_ID,Frame_ID,Global_Time,Local_X,Local_Y,Location",
        "v1,1,1000,1,1,I-80",
        "v2,1,1000,2,2,I-80",
        "v1,2,1100,1,1,I-80",
        "v2,2,1100,2,2,I-80",
        "v1,3,1200,1,1,I-80",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    inventory, internal = scan_ngsim_intervals(
        path,
        prefix_rows=3,
        chunksize=2,
    )
    assert internal["prefix_cutoff_keys"] == {("i_80", 1100)}
    assert inventory["runner_prefix_scope"]["partial_cutoff_frame_count"] == 1
    assert inventory["runner_prefix_scope"][
        "partial_cutoff_ranges_conservatively_invalid"
    ] == [
        {
            "source_segment_id": "i_80",
            "raw_time_start": 1100,
            "raw_time_end": 1100,
            "frame_count": 1,
            "sampling_interval": 1,
        }
    ]


def test_historical_registry_parse_duplicate_and_semantic_hash(tmp_path: Path) -> None:
    window = {
        "window_id": "legacy",
        "frame_offset": 0,
        "window_length": 24,
        "time_index_start": 2_000_000_000,
        "time_index_end": 2_000_002_300,
        "source_segment_id": "peachtree",
    }
    paths = []
    for name in ("train_window_plan.json", "formal_window_plan.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"selected_window_plan": [window], "total_reward": 999}), encoding="utf-8")
        paths.append(path)
    assert extract_selected_window_plan(paths[0]) == [window]
    registry, validation = build_historical_registry(
        ROOT,
        inventory_internal=synthetic_inventory_internal(),
        mobility_sha256="mobility",
        plan_paths=paths,
    )
    assert validation["passed"]
    assert registry["summary"]["raw_window_reference_count"] == 2
    assert registry["summary"]["unique_outer_interval_count"] == 1
    assert registry["summary"]["duplicate_reference_count"] == 1
    assert registry["records"][0]["consumed_status"] == "consumed_permanent"
    assert registry["hashes"]["semantic_sha256"] == canonical_sha256(semantic_projection(registry))


def test_unknown_historical_interval_has_conservative_exclusion(tmp_path: Path) -> None:
    path = tmp_path / "unknown_window_plan.json"
    path.write_text(
        json.dumps({"selected_window_plan": [{"window_id": "unknown", "frame_offset": 999, "window_length": 24}]}),
        encoding="utf-8",
    )
    registry, validation = build_historical_registry(
        ROOT,
        inventory_internal=synthetic_inventory_internal(),
        mobility_sha256="mobility",
        plan_paths=[path],
    )
    record = registry["records"][0]
    assert validation["checks"]["all_unknowns_have_conservative_scope"]
    assert record["unknown_interval_flag"]
    assert record["conservative_exclusion_scope"] == ["peachtree"]


def test_frame_time_and_segment_frame_overlap() -> None:
    left = make_window(0)
    right = deepcopy(left)
    right["window_id"] = "other"
    relation = interval_relation(left, right, minimum_gap_frames=24)
    assert relation["classification"] == "exact_overlap"
    assert relation["frame_overlap"] and relation["time_overlap"] and relation["segment_frame_overlap"]


def test_different_segment_run_is_safe_even_when_raw_frame_repeats() -> None:
    left = make_window(0, run="i_80_run_001")
    right = make_window(0, run="i_80_run_002", window_id="other-run")
    right["raw_time_start"] += 1_000_000
    right["raw_time_end"] += 1_000_000
    assert interval_relation(left, right, minimum_gap_frames=24)["classification"] == "safe"


def test_insufficient_gap_and_minimum_gap_boundary() -> None:
    left = make_window(0)
    near = make_window(1)
    near["raw_frame_start"] -= 1
    near["raw_frame_end"] -= 1
    near["raw_time_start"] -= 100
    near["raw_time_end"] -= 100
    assert interval_relation(left, near, minimum_gap_frames=24)["classification"] == "insufficient_gap"
    assert interval_relation(left, make_window(1), minimum_gap_frames=24)["classification"] == "safe"


def test_same_interval_different_id_and_mixed_full_outer_dedup() -> None:
    left = make_window(0, window_id="mixed")
    right = deepcopy(left)
    right["window_id"] = "full"
    assert interval_relation(left, right, minimum_gap_frames=24)["classification"] == "exact_overlap"
    identities = {
        canonical_sha256(
            {key: item[key] for key in ("source_segment_run_id", "raw_frame_start", "raw_frame_end", "raw_time_start", "raw_time_end")}
        )
        for item in (left, right)
    }
    assert len(identities) == 1


def test_seed_and_workflow_repetition_does_not_increase_outer_count(split_bundle: tuple[dict, dict, dict]) -> None:
    _, audit, _ = split_bundle
    assert audit["seed_workflow_repetitions_count_as_outer"] is False
    assert audit["outer_cluster_counts"] == {"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12}


def test_deterministic_split_and_split_hash(candidate_inventory: dict) -> None:
    first, _, _ = build_split_manifest(
        candidate_inventory,
        counts={"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12},
        minimum_gap_frames=24,
        created_at="2026-08-20T00:00:00+00:00",
    )
    second, _, _ = build_split_manifest(
        candidate_inventory,
        counts={"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12},
        minimum_gap_frames=24,
        created_at="2027-01-01T00:00:00+00:00",
    )
    assert [item["window_id"] for item in first["splits"]["formal"]["selected_window_plan"]] == [
        item["window_id"] for item in second["splits"]["formal"]["selected_window_plan"]
    ]
    assert first["hashes"]["semantic_sha256"] == second["hashes"]["semantic_sha256"]


def test_minimum_formal_and_holdout_counts_enforced(candidate_inventory: dict) -> None:
    with pytest.raises(InsufficientWindowError, match="formal<12"):
        build_split_manifest(
            candidate_inventory,
            counts={"train": 24, "dev": 12, "formal": 11, "sealed_holdout": 12},
            minimum_gap_frames=24,
        )
    with pytest.raises(InsufficientWindowError, match="sealed_holdout<12"):
        build_split_manifest(
            candidate_inventory,
            counts={"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 11},
            minimum_gap_frames=24,
        )


def test_insufficient_window_blocker_does_not_lower_requirement(candidate_inventory: dict) -> None:
    insufficient = deepcopy(candidate_inventory)
    insufficient["candidates"] = insufficient["candidates"][:23]
    with pytest.raises(InsufficientWindowError, match="BLOCKED_INSUFFICIENT_UNCONSUMED_WINDOWS"):
        build_split_manifest(
            insufficient,
            counts={"train": 0, "dev": 0, "formal": 12, "sealed_holdout": 12},
            minimum_gap_frames=24,
        )


def test_result_blind_split_builder_rejects_outcome_fields() -> None:
    with pytest.raises(FormalProtocolError, match="result-driven"):
        assert_result_blind_windows([{**make_window(0), "reward": 1.0}])


def test_protocol_canonical_hash_and_created_at_exclusion(protocol: dict) -> None:
    assert validate_protocol_manifest(protocol)["passed"]
    changed = deepcopy(protocol)
    changed["created_at"] = "2099-01-01T00:00:00+00:00"
    assert canonical_sha256(semantic_projection(changed)) == canonical_sha256(semantic_projection(protocol))
    assert canonical_json_bytes({"b": 1, "a": "中"}) == canonical_json_bytes({"a": "中", "b": 1})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("seed_plan.seeds.0", 5),
        ("training_budget.episodes_per_learned_agent_seed_capacity", 255),
        ("typed_catalog_and_capacity.capacity_strata.0.capacity_mb", 999.0),
        ("endpoints.primary.0", "different_metric"),
        ("statistics.multiplicity.method", "none"),
        ("claim_evidence_map.rows.0.claim_id", "changed"),
    ],
)
def test_every_semantic_protocol_family_changes_hash(protocol: dict, path: str, value: object) -> None:
    assert protocol_hash_changes_on_mutation(protocol, path, value)


def test_protocol_rejects_unknown_field_and_version(protocol: dict) -> None:
    unknown = deepcopy(protocol)
    unknown["surprise"] = True
    with pytest.raises(FormalProtocolError, match="unknown"):
        validate_protocol_manifest(unknown)
    bad_version = deepcopy(protocol)
    bad_version["typed_model_cache_formal_protocol_version"] = "2.0.0"
    with pytest.raises(FormalProtocolError, match="unsupported"):
        validate_protocol_manifest(bad_version)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_protocol_rejects_nan_and_infinity(protocol: dict, value: float) -> None:
    bad = deepcopy(protocol)
    bad["training_budget"]["learning_rates"]["ppo"] = value
    with pytest.raises(FormalProtocolError, match="non-finite"):
        validate_protocol_manifest(bad)


def test_agent_matrix_has_required_agents_and_oracles(protocol: dict) -> None:
    table = {item["agent"] for item in protocol["agent_matrix"]["controller_table"]}
    assert {"sa_ghmappo", "ppo", "mappo", "popularity_cache_heuristic", "cache_offload_drl"}.issubset(table)
    assert {item["horizon"] for item in protocol["agent_matrix"]["exact_oracle_cells"]} == {1, 3, 6, 12}
    assert len(protocol["agent_matrix"]["reactive_cache_policy_isolation"]) == 5


def test_seed_plan_and_equal_budget_are_frozen(protocol: dict) -> None:
    assert protocol["seed_plan"]["seeds"] == [7, 13, 29, 43, 71]
    budget = protocol["training_budget"]
    assert budget["maximum_environment_interactions_per_learned_agent_seed_capacity"] == 256 * 22
    assert budget["equal_budget_rule"].startswith("all learned agents")
    assert budget["early_stop"].startswith("disabled")


def test_capacity_strata_are_absolute_and_ordered(runtime_contract: dict) -> None:
    capacity = build_capacity_strata(runtime_contract)
    assert [(item["stratum"], item["capacity_mb"]) for item in capacity["strata"]] == [
        ("constrained", 288.0),
        ("medium", 576.0),
        ("relaxed", 864.0),
    ]
    assert capacity["workflow_state_counts_toward_long_term_capacity"] is False
    assert capacity["kv_prefix_enabled"] is False


def test_primary_endpoints_holm_family_and_claim_template(protocol: dict) -> None:
    assert set(protocol["endpoints"]["primary"]) == {
        "full_service_ready_byte_hit_rate",
        "joint_base_adapter_hit_rate",
        "full_service_ready_request_rate",
        "transfer_mb_per_request",
        "workflow_continuity_rate",
        "end_to_end_workflow_delay",
    }
    assert protocol["statistics"]["multiplicity"]["method"] == "Holm"
    statuses = protocol["claim_evidence_map"]["rows"][0]["allowed_result_statuses"]
    assert statuses == ["supported", "mixed", "unsupported", "contradicted", "unavailable"]
    assert all(item["result_status"] is None for item in protocol["claim_evidence_map"]["rows"])


def test_cli_semantic_override_rejected() -> None:
    assert_no_semantic_cli_overrides(["--output_dir", "/tmp/a", "--created_at=x"])
    with pytest.raises(FormalProtocolError, match="--seeds"):
        assert_no_semantic_cli_overrides(["--seeds", "7"])


def test_holdout_is_sealed_and_ordinary_runner_cannot_open(split_bundle: tuple[dict, dict, dict]) -> None:
    split, _, _ = split_bundle
    seal = build_holdout_seal(split)
    assert seal["sealed"] and not seal["opened"]
    assert seal["one_time_execution_token_status"] == "not_issued_in_G14B"
    with pytest.raises(HoldoutAccessError):
        validate_split_access("sealed_holdout", caller_role="benchmark_runner")


def test_holdout_one_time_open_and_append_only_record(split_bundle: tuple[dict, dict, dict], tmp_path: Path) -> None:
    split, _, _ = split_bundle
    seal = build_holdout_seal(split)
    token = "test-only-one-time-token"
    seal["one_time_execution_token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    gates = {name: True for name in seal["opening_gate"]["allowed_checks"]}
    log = tmp_path / "execution.jsonl"
    record = append_holdout_execution_record(
        log,
        seal_record=seal,
        execution_token=token,
        gate_results=gates,
        execution_commit="commit",
        command=["dedicated-executor"],
        output_run_id="run",
    )
    assert record["consumed_permanently"]
    with pytest.raises(HoldoutAccessError, match="already exists"):
        append_holdout_execution_record(
            log,
            seal_record=seal,
            execution_token=token,
            gate_results=gates,
            execution_commit="commit",
            command=["dedicated-executor"],
            output_run_id="run-2",
        )


def test_readiness_blocker_and_success() -> None:
    assert readiness_verdict({"split": True, "agents": False}) == "BLOCKED_G14B_READINESS_V2"
    assert readiness_verdict({"split": True, "agents": True}) == "READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL"


def test_json_round_trip_for_protocol_split_and_claims(protocol: dict, split_bundle: tuple[dict, dict, dict]) -> None:
    split, audit, allocation = split_bundle
    for payload in (protocol, split, audit, allocation["_pairwise_matrix"], build_claim_evidence_template(), build_statistics_protocol()):
        assert json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)) == payload
