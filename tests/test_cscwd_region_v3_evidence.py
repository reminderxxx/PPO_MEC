"""Synthetic-only checks for v3 identities, persistence and hard budget."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.confirm_cscwd_phase_v2_public_prepare_regions import step_once
from scripts.confirm_cscwd_phase_v2_public_prepare_regions_v3 import RUN_ID, run
from scripts.cscwd_region_v3_evidence import (
    BRANCH_RECORD_VERSION, atomic_record, classify_identities,
    execute_and_persist, partial_evidence_status, read_verified_record, state_identities,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config
from src.envs.core.raw_ngsim_event_time_env import RawVehicleTrace
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env


ROOT = Path(__file__).resolve().parents[1]


def synthetic_env(suffix: str = "forward") -> RawNGSIMPhaseV2Env:
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    contract = json.loads((ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json").read_text())
    workload = json.loads((
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instance = next(deepcopy(row) for row in workload["instances"] if row["design_id"] == "dev_01")
    times = tuple(round(index * 0.1, 9) for index in range(120))
    positions = [(0.0, 7.4 * time) for time in times]
    if suffix == "reverse":
        positions[2:] = [(0.0, -7.4 * time) for time in times[2:]]
    trace = RawVehicleTrace(times, tuple(positions), "fixture_vehicle")
    return RawNGSIMPhaseV2Env(config, instance, trace,
                              phase_contract=contract["physical_assumptions"],
                              condition=contract["conditions"][0])


def test_same_public_observation_with_different_exogenous_future_is_kept() -> None:
    forward, reverse = synthetic_env(), synthetic_env("reverse")
    _, a = forward.reset()
    _, b = reverse.reset()
    first, second = state_identities(forward, a), state_identities(reverse, b)
    assert first["public_observation_sha256"] == second["public_observation_sha256"]
    assert first["full_simulation_sha256"] != second["full_simulation_sha256"]
    rows = classify_identities([{"start_id": "a", **first}, {"start_id": "b", **second}])
    assert [row["full_duplicate_of"] for row in rows] == [None, None]
    assert all(row["same_public_different_full"] for row in rows)
    assert all(row["public_group_distinct_full_state_count"] == 2 for row in rows)


def test_exact_full_duplicate_is_alias_but_prefix_change_is_distinct() -> None:
    env = synthetic_env()
    _, info = env.reset()
    before = state_identities(env, info)
    exact = state_identities(deepcopy(env), deepcopy(info))
    assert before == exact
    rows = classify_identities([{"start_id": "a", **before}, {"start_id": "b", **exact}])
    assert rows[1]["full_duplicate_of"] == "a"
    action = next(index for index, allowed in enumerate(info["action_mask"]) if allowed)
    _, _, _, _, after_info = env.step(action)
    after = state_identities(env, after_info)
    assert after["full_simulation_sha256"] != before["full_simulation_sha256"]
    rows = classify_identities([{"start_id": "a", **before}, {"start_id": "after_prefix", **after}])
    assert rows[1]["full_duplicate_of"] is None


def test_completed_branch_survives_next_branch_failure_and_hash_is_checked(tmp_path: Path) -> None:
    tasks = [{"start_id_safe": "s00", "action": 3}, {"start_id_safe": "s00", "action": 4}]

    def execute(task):
        if task["action"] == 4:
            raise RuntimeError("injected middle-branch failure")
        return {"input_identity": {"full_simulation_sha256": "abc"},
                "events": [{"step": 0}], "metrics": {"workflow_completed": True}}

    with pytest.raises(RuntimeError, match="injected"):
        execute_and_persist(tasks, execute, tmp_path / "branches")
    complete = tmp_path / "branches/s00_action3.json"
    assert read_verified_record(complete, version=BRANCH_RECORD_VERSION)["events"] == [{"step": 0}]
    assert not (tmp_path / "branches/s00_action4.json").exists()
    partial = partial_evidence_status(tmp_path / "branches", current_branch="s00:action4",
                                      environment_step_count=7)
    assert partial["partial_verdict"] == "PARTIAL_UNVERIFIABLE"
    assert partial["completed_branch_count"] == 1
    assert partial["completed_branch_files"] == ["s00_action3.json"]
    assert partial["incomplete_branch"] == "s00:action4"
    data = json.loads(complete.read_text())
    data["payload"]["events"][0]["step"] = 99
    complete.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        read_verified_record(complete, version=BRANCH_RECORD_VERSION)
    corrupt = partial_evidence_status(tmp_path / "branches", current_branch=None,
                                      environment_step_count=7)
    assert corrupt["completed_branch_count"] == 0
    assert corrupt["corrupt_or_unverifiable_branch_files"] == ["s00_action3.json"]


def test_atomic_record_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    atomic_record(path, {"rows": []}, version="fixture_v1")
    with pytest.raises(FileExistsError):
        atomic_record(path, {"rows": [1]}, version="fixture_v1")
    assert read_verified_record(path, version="fixture_v1") == {"rows": []}


def test_budget_rejects_step_before_env_call() -> None:
    class Trap:
        def step(self, action):
            raise AssertionError("must not call env.step")

    with pytest.raises(RuntimeError, match="hard cap"):
        step_once(Trap(), 3, {"action_mask": [True] * 5},
                  start_id="synthetic", condition_id="default", branch="action3",
                  budget={"env_steps": 288})


def test_python39_entry_identity() -> None:
    assert RUN_ID.endswith("_v3")


def test_unfrozen_output_root_rejected_before_input_reads(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="frozen run identity"):
        run(tmp_path / "missing_protocol", tmp_path / "missing_pilot",
            tmp_path / "wrong_root", tmp_path / "missing_authorization")
