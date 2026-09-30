import json
from pathlib import Path

import pytest

import scripts.calibrate_workflow_state_recovery as calibration
from scripts.synthesize_adapter_state_recovery_calibration import synthesize
from scripts.calibrate_workflow_state_recovery import (
    add_state_integrity,
    canonical_bytes,
    durable_create,
    normalize_status,
    pair_correctness,
    state_without_integrity,
    validate_state,
    verify_plan,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/analysis/adapter_state_recovery_calibration_20260930_v1"
PLAN_PATH = RUN_ROOT / "measurement_plan.json"
INVENTORY_PATH = RUN_ROOT / "adapter_resource_inventory.json"


def plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def stage(text=" Clear.", token_ids=None):
    return {
        "decoded_text": text,
        "generated_token_ids": token_ids or [1200, 13],
        "normalized_status": normalize_status(text),
    }


def resources():
    return {"tokenizer.json": {"bytes": 4, "sha256": "a" * 64}}


def test_frozen_plan_and_resource_authorization_boundary():
    value = plan()
    verify_plan(value)
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inventory["local_inventory"]["adapter_config_paths"] == []
    assert inventory["authorization_boundary"]["new_weight_download_authorized"] is False
    assert inventory["minimum_new_weight_budget"]["adapter_weight_bytes"] == 164065376
    assert [row["declared_base_model"] for row in inventory["candidate_pair"]] == [
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "HuggingFaceTB/SmolVLM-500M-Instruct",
    ]
    assert all(row["task_type"] is None for row in inventory["candidate_pair"])


def test_state_contract_exact_round_trip_and_tamper_rejection(tmp_path):
    value = plan()
    payload = state_without_integrity(value, stage(), resources(), 1)
    state = add_state_integrity(payload)
    validate_state(state, value, resources())
    path = tmp_path / "state.json"
    durable_create(path, canonical_bytes(state))
    assert json.loads(path.read_bytes()) == state

    tampered = json.loads(path.read_bytes())
    tampered["node_outputs"]["normalize_event"]["normalized_status"] = "blocked"
    with pytest.raises(ValueError, match="content hash"):
        validate_state(tampered, value, resources())


def test_state_contract_rejects_resource_and_dag_drift():
    value = plan()
    state = add_state_integrity(state_without_integrity(value, stage(), resources(), 1))
    with pytest.raises(ValueError, match="processor resource"):
        validate_state(
            state,
            value,
            {"tokenizer.json": {"bytes": 5, "sha256": "b" * 64}},
        )
    drifted = {key: item for key, item in state.items() if key != "integrity"}
    drifted["dag"] = {"execution_order": ["publish_event"], "edges": []}
    drifted = add_state_integrity(drifted)
    with pytest.raises(ValueError, match="DAG"):
        validate_state(drifted, value, resources())


def test_pair_correctness_requires_process_order_state_and_outputs():
    value = plan()
    continuous = {
        "status": "pass",
        "pid": 10,
        "stage1": stage(),
        "stage2": stage(" clear", [1200]),
    }
    source = {
        "status": "pass",
        "pid": 11,
        "finished_unix_ns": 20,
        "stage1": stage(),
        "state": {"sha256": "c" * 64, "bytes": 100},
    }
    target = {
        "status": "pass",
        "pid": 12,
        "started_unix_ns": 21,
        "stage2": stage(" clear", [1200]),
        "state": {"sha256": "c" * 64, "bytes": 100},
    }
    result = pair_correctness(
        value, {"continuous": continuous, "source": source, "target": target}
    )
    assert result["pair_pass"] is True
    target["stage2"] = stage(" blocked", [999])
    result = pair_correctness(
        value, {"continuous": continuous, "source": source, "target": target}
    )
    assert result["pair_pass"] is False
    assert result["checks"]["final_token_ids_exact"] is False


def test_normalization_and_create_only_state(tmp_path):
    assert normalize_status(" Clear.\n") == "clear"
    path = tmp_path / "state.json"
    durable_create(path, b"{}")
    with pytest.raises(FileExistsError):
        durable_create(path, b"{}")


def test_orchestrator_persists_failed_warmup(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "verify_base",
        lambda _plan: {"path": "fixture", "bytes": 1, "sha256": "a" * 64, "match": True},
    )
    monkeypatch.setattr(
        calibration,
        "run_pair",
        lambda *_args, **_kwargs: {
            "measurement_index": -1,
            "rows": {
                "continuous": {
                    "status": "failed",
                    "mode": "continuous",
                    "returncode": 1,
                    "error_type": "FixtureError",
                    "error": "bounded failure",
                    "stderr": "fixture",
                }
            },
            "correctness": {"pair_pass": False},
        },
    )
    output_path = tmp_path / "raw.json"
    state_root = tmp_path / "states"
    result = calibration.orchestrate(PLAN_PATH, output_path, state_root)
    assert output_path.is_file()
    assert state_root.exists() is False
    assert result["measurement_count"] == 0
    assert result["all_measurements_pass"] is False
    assert result["failure_records"][0]["error_type"] == "FixtureError"


def test_layered_synthesis_preserves_native_ledger_and_failed_calibration():
    output = synthesize(
        ROOT
        / "artifacts/analysis/remaining_workflow_decision_value_audit_20260930_v1/paired_witness_results.json",
        RUN_ROOT / "workflow_state_recovery_measurements.json",
        INVENTORY_PATH,
        PLAN_PATH,
    )
    native = output["native_synthetic_ledger_unchanged"]
    assert len(native) == 2
    assert all(row["avoided_synthetic_model_transfer_bytes"] == 104 * 1024 * 1024 for row in native)
    assert all(row["equal_completion"] and row["equal_failures"] for row in native)
    assert output["workflow_state_recovery"]["independent_process_restore_witness"] is False
    assert output["workflow_state_recovery"]["source_state_bytes_diagnostic_only"] == 3085
    assert output["adapter_compatibility"]["status"] == "unavailable"
    assert output["calibration_and_sensitivity"]["real_model_byte_substitution_performed"] is False
    assert len(output["calibration_and_sensitivity"]["finite_assumption_grid"]) == 9
    assert output["decision_effect"]["next_small_scale_method_comparison_ready"] is False
