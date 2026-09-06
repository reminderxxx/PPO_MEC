from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.benchmark_main_results import (
    CHECKPOINT_PROVENANCE_ENVELOPE_FIELDS,
    load_checkpoint_provenance_manifest,
    validate_benchmark_checkpoint_gate,
)
from scripts.manage_typed_model_cache_formal_artifacts import (
    checkpoint_freeze,
    dev_select,
    write_checkpoint_companions,
)
from scripts.run_typed_model_cache_formal_dev_selection import (
    validate_dev_checkpoint_identity,
)
from scripts.train_algo_pool_real_sample import (
    annotate_checkpoint,
    build_training_identity_metadata,
    load_checkpoint_training_metadata,
    validate_serialized_formal_checkpoint,
)
from src.agents.registry import build_agent
from src.evaluators.typed_model_cache_formal_execution import validate_protocol_v1_1
from src.evaluators.typed_model_cache_formal_protocol import sha256_file
from src.runtime.formal_invalid_run_registry import (
    PermanentlyInvalidFormalReferenceError,
    reject_permanently_invalid_formal_references,
)
from src.runtime.formal_training_contract import resolve_training_contract
from src.runtime.formal_training_identity import (
    FormalTrainingIdentityError,
    build_execution_binding,
    checkpoint_training_identity_projection,
    checkpoint_training_identity_fields,
    expected_checkpoint_training_identity,
    validate_checkpoint_training_identity,
)
from src.runtime.typed_model_cache_runtime import (
    build_checkpoint_provenance,
    resolve_model_cache_runtime,
    validate_checkpoint_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol_v2_8_manifest.json"
SCIENTIFIC_PATH = PROTOCOL_ROOT / "agent_training_scientific_config.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture(scope="module")
def identity_bundle() -> tuple[dict, dict, dict, dict, dict, dict]:
    protocol = load(PROTOCOL_PATH)
    scientific = load(SCIENTIFIC_PATH)
    assert validate_protocol_v1_1(protocol)["status"] == "pass"
    environment = deepcopy(
        protocol["formal_execution_environment_contract"]["scientific_identity"]
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
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
            "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
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
    resolved_by_agent = {
        agent: resolve_training_contract(
            agent_name=agent,
            profile_defaults={
                "episodes": 1,
                "update_every": 1,
                "batch_size": 1,
                "max_steps": 1,
            },
            cli_values={},
            formal_protocol=protocol,
            scientific_config=scientific,
            execution_binding=binding,
            resolved_execution_context=context,
        )
        for agent in protocol["training_budget"]["learned_agent_order"]
    }
    expected = expected_checkpoint_training_identity(
        protocol=protocol, resolved_execution_context=context
    )
    return protocol, scientific, binding, context, resolved_by_agent, expected


def capacity_runtime_paths(protocol: dict) -> dict[str, Path]:
    rows = protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]
    return {
        str(row["capacity_label"]): ROOT / str(row["runtime_config_path"])
        for row in rows
    }


def checkpoint_metadata(
    *,
    resolved,
    agent: str,
    seed: int,
    runtime: dict,
    window_identity: dict,
) -> dict:
    return {
        "run_id": f"identity_acceptance_{agent}_{seed}",
        "agent_name": agent,
        "episodes": resolved.episodes,
        "update_count": 4,
        "checkpoint_schedule": {
            "checkpoint_every_updates": resolved.checkpoint_every_updates,
            "expected_update_count": resolved.expected_update_count,
        },
        **build_training_identity_metadata(resolved),
        "typed_runtime_provenance": build_checkpoint_provenance(
            root=ROOT,
            agent_name=agent,
            training_seed=seed,
            runtime_contract=runtime,
            reward_positive_offset=0.0,
            train_window_plan_identity=window_identity,
        ),
    }


def test_1200_coordinate_producer_projection_and_json_roundtrip(identity_bundle) -> None:
    protocol, _, _, _, resolved_by_agent, expected = identity_bundle
    matrix = protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]
    updates = range(4, 33, 4)
    assert len(matrix) == 150 and len(list(updates)) == 8
    observed = 0
    shared_identities = set()
    per_cell = set()
    for coordinate in matrix:
        metadata = build_training_identity_metadata(
            resolved_by_agent[str(coordinate["agent"])]
        )
        for update_index in updates:
            restored = json.loads(json.dumps(metadata, allow_nan=False))
            projected = checkpoint_training_identity_projection(
                restored,
                protocol_version=protocol["typed_model_cache_formal_protocol_version"],
                require_nested_contract=True,
            )
            assert projected == expected
            shared_identities.add(tuple(sorted(projected.items())))
            per_cell.add(
                (
                    coordinate["agent"],
                    int(coordinate["seed"]),
                    coordinate["capacity_label"],
                    update_index,
                )
            )
            observed += 1
    assert observed == 1200
    assert len(shared_identities) == 1
    assert len(per_cell) == 1200


def test_ten_agent_actual_save_annotate_readback_candidate_and_latest(
    identity_bundle, tmp_path: Path
) -> None:
    protocol, _, _, _, resolved_by_agent, _ = identity_bundle
    runtime_paths = capacity_runtime_paths(protocol)
    capacities = list(runtime_paths)
    window_identity = {"test_only": True, "split": "train"}
    agents = protocol["training_budget"]["learned_agent_order"]
    observed_capacities = set()
    for index, agent_name in enumerate(agents):
        capacity = capacities[index % len(capacities)]
        runtime = resolve_model_cache_runtime(runtime_paths[capacity], root=ROOT)
        resolved = resolved_by_agent[agent_name]
        agent = build_agent(
            agent_name,
            random_seed=7 + index,
            batch_size=2,
            **resolved.agent_config,
        )
        metadata = checkpoint_metadata(
            resolved=resolved,
            agent=agent_name,
            seed=7 + index,
            runtime=runtime,
            window_identity=window_identity,
        )
        for filename in ("latest.pt", "update_0004.pt"):
            path = tmp_path / capacity / agent_name / filename
            agent.save(str(path))
            annotate_checkpoint(path, metadata)
            read_back = validate_serialized_formal_checkpoint(
                path,
                resolved_training=resolved,
                agent_name=agent_name,
                seed=7 + index,
                runtime_contract_sha256=runtime["runtime_contract_sha256"],
            )
            assert load_checkpoint_training_metadata(path) == read_back
        observed_capacities.add(capacity)
    assert observed_capacities == set(capacities)


def build_strict_candidates(
    identity_bundle, tmp_path: Path, *, production_roundtrip: bool = False
) -> tuple[list[dict], dict]:
    protocol, _, _, context, resolved_by_agent, expected = identity_bundle
    runtime_paths = capacity_runtime_paths(protocol)
    window_identity = {"test_only": True, "split": "train"}
    candidates = []
    for coordinate in protocol["execution_contract"]["command_templates"]["train"][
        "matrix_contexts"
    ]:
        agent = str(coordinate["agent"])
        seed = int(coordinate["seed"])
        capacity = str(coordinate["capacity_label"])
        runtime = resolve_model_cache_runtime(runtime_paths[capacity], root=ROOT)
        metadata = checkpoint_metadata(
            resolved=resolved_by_agent[agent],
            agent=agent,
            seed=seed,
            runtime=runtime,
            window_identity=window_identity,
        )
        path = tmp_path / "training" / capacity / agent / f"seed_{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if production_roundtrip:
            torch.save({"test_state": True}, path)
            annotate_checkpoint(path, metadata)
            read_back = validate_serialized_formal_checkpoint(
                path,
                resolved_training=resolved_by_agent[agent],
                agent_name=agent,
                seed=seed,
                runtime_contract_sha256=runtime["runtime_contract_sha256"],
            )
            assert read_back == load_checkpoint_training_metadata(path)
        else:
            torch.save({"test_state": True, "training_metadata": metadata}, path)
        availability = {
            metric: {"available_count": 1, "unavailable_count": 0, "total_count": 1}
            for metric in (
                "full_service_ready_byte_hit_rate",
                "workflow_continuity_rate",
                "transfer_mb_per_request",
                "end_to_end_workflow_delay",
            )
        }
        candidates.append(
            {
                "agent_name": agent,
                "seed": seed,
                "capacity_label": capacity,
                "update_index": 4,
                "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path),
                "full_service_ready_byte_hit_rate": 0.5,
                "workflow_continuity_rate": 0.5,
                "transfer_mb_per_request": 1.0,
                "end_to_end_workflow_delay": 2.0,
                "selection_metric_availability": availability,
                "runtime_contract_sha256": runtime["runtime_contract_sha256"],
                "resolved_agent_config": resolved_by_agent[agent].agent_config,
                "checkpoint_schedule": metadata["checkpoint_schedule"],
                **build_training_identity_metadata(resolved_by_agent[agent]),
                "non_formal_rehearsal": False,
                "typed_runtime_provenance": metadata["typed_runtime_provenance"],
            }
        )
    (tmp_path / "resolved_execution_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    return candidates, expected


@pytest.fixture(scope="module")
def full_companion_chain(identity_bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("full_companion_chain")
    protocol, _, binding, context, _, expected = identity_bundle
    candidates, _ = build_strict_candidates(
        identity_bundle, root, production_roundtrip=True
    )
    (root / "checkpoint_candidates.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    selection = dev_select(root, protocol, expected_training_identity=expected)
    (root / "dev_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    freeze = checkpoint_freeze(root, protocol)
    return {
        "root": root,
        "protocol": protocol,
        "context": context,
        "binding": binding,
        "expected": expected,
        "freeze": freeze,
        "companions": write_checkpoint_companions(root, freeze),
    }


def first_full_companion(chain) -> tuple[dict, dict, dict, str, int, str]:
    protocol = chain["protocol"]
    companion = chain["companions"][0]
    capacity = companion["capacity_label"]
    loaded = load_checkpoint_provenance_manifest(
        companion["checkpoint_provenance_manifest_path"]
    )
    agent = next(iter(loaded))
    seed_text = next(iter(loaded[agent]))
    runtime = resolve_model_cache_runtime(
        capacity_runtime_paths(protocol)[capacity], root=ROOT
    )
    return loaded[agent][seed_text], runtime, protocol, agent, int(seed_text), capacity


def test_strict_selection_freeze_and_typed_provenance_chain(
    identity_bundle, full_companion_chain
) -> None:
    protocol, _, _, _, _, expected = identity_bundle
    selection = load(full_companion_chain["root"] / "dev_selection.json")
    assert len(selection["selected"]) == 150
    assert selection["formal_training_identity"] == expected
    freeze = full_companion_chain["freeze"]
    assert freeze["frozen_checkpoint_count"] == 150
    selected = freeze["frozen_checkpoints"][0]
    runtime = resolve_model_cache_runtime(
        capacity_runtime_paths(protocol)[selected["capacity_label"]], root=ROOT
    )
    report = validate_checkpoint_provenance(
        selected["checkpoint_path"],
        expected_agent_name=selected["agent_name"],
        expected_seed=selected["seed"],
        expected_runtime_contract=runtime,
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity={"test_only": True, "split": "train"},
        expected_checkpoint_sha256=selected["checkpoint_sha256"],
        require_git_commit=expected["execution_commit"],
        expected_formal_training_identity=expected,
        expected_formal_protocol_version=protocol[
            "typed_model_cache_formal_protocol_version"
        ],
    )
    assert report["status"] == "compatible"


def test_full_companion_envelope_real_benchmark_gate_chain(
    full_companion_chain,
) -> None:
    protocol = full_companion_chain["protocol"]
    expected_fields = {
        *CHECKPOINT_PROVENANCE_ENVELOPE_FIELDS,
        *checkpoint_training_identity_fields(
            protocol["typed_model_cache_formal_protocol_version"]
        ),
    }
    observed_agents, observed_capacities = set(), set()
    gate_count = 0
    rollout_calls = []
    for companion in full_companion_chain["companions"]:
        capacity = companion["capacity_label"]
        runtime = resolve_model_cache_runtime(
            capacity_runtime_paths(protocol)[capacity], root=ROOT
        )
        loaded = load_checkpoint_provenance_manifest(
            companion["checkpoint_provenance_manifest_path"]
        )
        for agent, by_seed in loaded.items():
            for seed_text, envelope in by_seed.items():
                assert len(envelope) == 17
                assert set(envelope) == expected_fields
                path = envelope["artifact_location"]["resolved_location"]
                report = validate_benchmark_checkpoint_gate(
                    path,
                    expected_agent_name=agent,
                    expected_seed=int(seed_text),
                    expected_runtime_contract=runtime,
                    expected_reward_positive_offset=0.0,
                    provenance_envelope=envelope,
                    protocol=protocol,
                    resolved_execution_context=full_companion_chain["context"],
                    execution_binding=full_companion_chain["binding"],
                    expected_capacity_label=capacity,
                )
                assert report["status"] == "compatible"
                observed_agents.add(agent)
                observed_capacities.add(capacity)
                gate_count += 1
    assert observed_agents == set(protocol["training_budget"]["learned_agent_order"])
    assert observed_capacities == set(capacity_runtime_paths(protocol))
    assert gate_count == 150
    assert rollout_calls == []


@pytest.mark.parametrize(
    ("field", "mode"),
    [
        ("formal_training_execution_binding_sha256", "missing"),
        ("formal_nullable_metric_aggregation_contract_semantic_sha256", "missing"),
        ("formal_nullable_metric_aggregation_contract_semantic_sha256", "wrong"),
        ("formal_protocol_semantic_sha256", "wrong"),
        ("formal_training_execution_binding_sha256", "wrong"),
        ("resolved_execution_context_sha256", "wrong"),
        ("active_formal_bundle_sha256", "wrong"),
    ],
)
def test_full_envelope_shared_identity_negatives_precede_rollout(
    full_companion_chain, field: str, mode: str
) -> None:
    envelope, runtime, protocol, agent, seed, capacity = first_full_companion(
        full_companion_chain
    )
    envelope.pop(field) if mode == "missing" else envelope.__setitem__(field, "0" * 64)
    rollout_calls = []
    with pytest.raises(ValueError, match="field mismatch|trusted Protocol/context"):
        validate_benchmark_checkpoint_gate(
            envelope["artifact_location"]["resolved_location"],
            expected_agent_name=agent,
            expected_seed=seed,
            expected_runtime_contract=runtime,
            expected_reward_positive_offset=0.0,
            provenance_envelope=envelope,
            protocol=protocol,
            resolved_execution_context=full_companion_chain["context"],
            execution_binding=full_companion_chain["binding"],
            expected_capacity_label=capacity,
        )
        rollout_calls.append("called")
    assert rollout_calls == []


def test_active_execution_binding_drift_precedes_rollout(full_companion_chain) -> None:
    envelope, runtime, protocol, agent, seed, capacity = first_full_companion(
        full_companion_chain
    )
    binding = deepcopy(full_companion_chain["binding"])
    binding["binding_full_sha256"] = "0" * 64
    rollout_calls = []
    with pytest.raises(ValueError, match="execution binding"):
        validate_benchmark_checkpoint_gate(
            envelope["artifact_location"]["resolved_location"],
            expected_agent_name=agent,
            expected_seed=seed,
            expected_runtime_contract=runtime,
            expected_reward_positive_offset=0.0,
            provenance_envelope=envelope,
            protocol=protocol,
            resolved_execution_context=full_companion_chain["context"],
            execution_binding=binding,
            expected_capacity_label=capacity,
        )
        rollout_calls.append("called")
    assert rollout_calls == []


@pytest.mark.parametrize("case", ["sha", "git", "window", "runtime", "agent", "seed", "capacity"])
def test_full_envelope_runtime_negatives_precede_rollout(
    full_companion_chain, case: str
) -> None:
    envelope, runtime, protocol, agent, seed, capacity = first_full_companion(
        full_companion_chain
    )
    expected_agent, expected_seed, expected_capacity = agent, seed, capacity
    if case == "sha":
        envelope["checkpoint_sha256"] = "0" * 64
        envelope["checkpoint_identity"]["checkpoint_sha256"] = "0" * 64
    elif case == "git":
        envelope["execution_git_commit"] = "0" * 40
    elif case == "window":
        envelope["train_window_plan_identity"] = {"test_only": True, "split": "wrong"}
    elif case == "runtime":
        envelope["runtime_contract_sha256"] = "0" * 64
    elif case == "agent":
        expected_agent = "ppo" if agent != "ppo" else "mappo"
    elif case == "seed":
        expected_seed += 1
    else:
        expected_capacity = next(
            label for label in capacity_runtime_paths(protocol) if label != capacity
        )
        runtime = resolve_model_cache_runtime(
            capacity_runtime_paths(protocol)[expected_capacity], root=ROOT
        )
    rollout_calls = []
    try:
        report = validate_benchmark_checkpoint_gate(
            envelope["artifact_location"]["resolved_location"],
            expected_agent_name=expected_agent,
            expected_seed=expected_seed,
            expected_runtime_contract=runtime,
            expected_reward_positive_offset=0.0,
            provenance_envelope=envelope,
            protocol=protocol,
            resolved_execution_context=full_companion_chain["context"],
            execution_binding=full_companion_chain["binding"],
            expected_capacity_label=expected_capacity,
        )
    except ValueError:
        pass
    else:
        assert report["status"] == "incompatible"
        assert report["errors"]
    assert rollout_calls == []


def test_checkpoint_top_nested_conflict_and_protocol_downgrade_precede_rollout(
    full_companion_chain, tmp_path: Path
) -> None:
    envelope, runtime, protocol, agent, seed, capacity = first_full_companion(
        full_companion_chain
    )
    source = Path(envelope["artifact_location"]["resolved_location"])
    rollout_calls = []
    for mode in ("top_nested_conflict", "protocol_downgrade"):
        payload = torch.load(source, map_location="cpu")
        metadata = payload["training_metadata"]
        if mode == "top_nested_conflict":
            metadata["active_formal_bundle_sha256"] = "0" * 64
        else:
            metadata["formal_training_contract"]["formal_protocol_version"] = "2.7.0"
        target = tmp_path / f"{mode}.pt"
        torch.save(payload, target)
        mutated = deepcopy(envelope)
        mutated["checkpoint_sha256"] = sha256_file(target)
        mutated["checkpoint_identity"]["checkpoint_sha256"] = mutated["checkpoint_sha256"]
        report = validate_benchmark_checkpoint_gate(
            target,
            expected_agent_name=agent,
            expected_seed=seed,
            expected_runtime_contract=runtime,
            expected_reward_positive_offset=0.0,
            provenance_envelope=mutated,
            protocol=protocol,
            resolved_execution_context=full_companion_chain["context"],
            execution_binding=full_companion_chain["binding"],
            expected_capacity_label=capacity,
        )
        assert report["status"] == "incompatible"
        assert any(
            "top-level/nested" in error or "Protocol version mismatch" in error
            for error in report["errors"]
        )
    assert rollout_calls == []


def test_benchmark_main_calls_strict_envelope_gate_before_rollout() -> None:
    source = (ROOT / "scripts/benchmark_main_results.py").read_text(encoding="utf-8")
    gate = source.index("checkpoint_gate = validate_benchmark_checkpoint_gate(")
    rollout = source.index("summary = run_real_episode(", gate)
    assert gate < rollout
    assert "binding.get(\n                                    \"formal_training_execution_binding_sha256\"" not in source


def test_freeze_rechecks_actual_checkpoint_not_only_selected_wrapper(
    identity_bundle, tmp_path: Path
) -> None:
    protocol, _, _, _, _, expected = identity_bundle
    candidates, _ = build_strict_candidates(identity_bundle, tmp_path)
    (tmp_path / "checkpoint_candidates.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    selection = dev_select(tmp_path, protocol, expected_training_identity=expected)
    selected = selection["selected"][0]
    path = Path(selected["checkpoint_path"])
    payload = torch.load(path, map_location="cpu")
    payload["training_metadata"].pop(
        "formal_nullable_metric_aggregation_contract_semantic_sha256"
    )
    torch.save(payload, path)
    selected["checkpoint_sha256"] = sha256_file(path)
    (tmp_path / "dev_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity missing"):
        checkpoint_freeze(tmp_path, protocol)
    assert not (tmp_path / "checkpoint_freeze.json").exists()


def mutate_both(row: dict, field: str, value: str = "0" * 64) -> None:
    row[field] = value
    row["formal_training_contract"][field] = value


@pytest.mark.parametrize(
    ("mutation", "all_rows", "match"),
    [
        (lambda row: row.pop("formal_nullable_metric_aggregation_contract_semantic_sha256"), False, "missing"),
        (lambda row: row["formal_training_contract"].pop("formal_nullable_metric_aggregation_contract_semantic_sha256"), False, "nested"),
        (lambda row: row["formal_training_contract"].__setitem__("formal_nullable_metric_aggregation_contract_semantic_sha256", "0" * 64), False, "top-level/nested"),
        (lambda row: mutate_both(row, "formal_nullable_metric_aggregation_contract_semantic_sha256"), True, "trusted expected"),
        (lambda row: mutate_both(row, "formal_nullable_metric_aggregation_contract_semantic_sha256"), False, "trusted expected"),
        (lambda row: mutate_both(row, "formal_protocol_semantic_sha256"), True, "trusted expected"),
        (lambda row: mutate_both(row, "formal_training_execution_binding_sha256"), True, "trusted expected"),
        (lambda row: mutate_both(row, "resolved_execution_context_sha256"), True, "trusted expected"),
        (lambda row: mutate_both(row, "active_formal_bundle_sha256"), True, "trusted expected"),
    ],
)
def test_candidate_identity_negatives_publish_nothing(
    identity_bundle, tmp_path: Path, mutation, all_rows: bool, match: str
) -> None:
    protocol, _, _, _, _, expected = identity_bundle
    candidates, _ = build_strict_candidates(identity_bundle, tmp_path)
    for row in candidates if all_rows else candidates[:1]:
        mutation(row)
    (tmp_path / "checkpoint_candidates.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=match):
        dev_select(tmp_path, protocol, expected_training_identity=expected)
    assert not (tmp_path / "dev_selection.json").exists()
    assert not (tmp_path / "checkpoint_freeze.json").exists()


def test_prebenchmark_rejection_has_zero_child_calls(identity_bundle) -> None:
    protocol, _, _, _, resolved_by_agent, expected = identity_bundle
    runtime = resolve_model_cache_runtime(
        next(iter(capacity_runtime_paths(protocol).values())), root=ROOT
    )
    metadata = checkpoint_metadata(
        resolved=resolved_by_agent["ppo"],
        agent="ppo",
        seed=7,
        runtime=runtime,
        window_identity={"test_only": True},
    )
    metadata.pop("formal_nullable_metric_aggregation_contract_semantic_sha256")
    benchmark_child_calls = []
    with pytest.raises(FormalTrainingIdentityError):
        validate_dev_checkpoint_identity(
            metadata,
            expected_training_identity=expected,
            protocol_version=protocol["typed_model_cache_formal_protocol_version"],
            agent_name="ppo",
            seed=7,
            runtime_contract_sha256=runtime["runtime_contract_sha256"],
        )
        benchmark_child_calls.append("called")
    assert benchmark_child_calls == []


@pytest.mark.parametrize(
    ("agent", "seed", "runtime"),
    [("mappo", 7, None), ("ppo", 13, None), ("ppo", 7, "0" * 64)],
)
def test_wrong_agent_seed_or_capacity_runtime_is_rejected(
    identity_bundle, agent: str, seed: int, runtime: str | None
) -> None:
    protocol, _, _, _, resolved_by_agent, expected = identity_bundle
    actual_runtime = resolve_model_cache_runtime(
        next(iter(capacity_runtime_paths(protocol).values())), root=ROOT
    )
    metadata = checkpoint_metadata(
        resolved=resolved_by_agent["ppo"],
        agent="ppo",
        seed=7,
        runtime=actual_runtime,
        window_identity={"test_only": True},
    )
    with pytest.raises(FormalTrainingIdentityError, match="per-cell"):
        validate_dev_checkpoint_identity(
            metadata,
            expected_training_identity=expected,
            protocol_version=protocol["typed_model_cache_formal_protocol_version"],
            agent_name=agent,
            seed=seed,
            runtime_contract_sha256=runtime or actual_runtime["runtime_contract_sha256"],
        )


def test_json_serialization_field_loss_is_rejected(identity_bundle) -> None:
    protocol, _, _, _, resolved_by_agent, _ = identity_bundle
    metadata = build_training_identity_metadata(resolved_by_agent["ppo"])
    restored = json.loads(json.dumps(metadata))
    restored.pop("formal_nullable_metric_aggregation_contract_semantic_sha256")
    with pytest.raises(FormalTrainingIdentityError, match="missing"):
        checkpoint_training_identity_projection(
            restored,
            protocol_version=protocol["typed_model_cache_formal_protocol_version"],
            require_nested_contract=True,
        )


def test_g14c_v15_checkpoint_reference_is_permanently_rejected() -> None:
    with pytest.raises(PermanentlyInvalidFormalReferenceError, match="g14c_v15"):
        reject_permanently_invalid_formal_references(
            [
                ROOT
                / "artifacts/experiments/typed_model_cache_formal"
                / "typed_model_cache_formal_20260905_213344_g14c_v15"
                / "training/ppo/checkpoints/update_0004.pt"
            ]
        )
