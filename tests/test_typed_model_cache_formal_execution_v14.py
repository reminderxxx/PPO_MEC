from __future__ import annotations

import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.manage_typed_model_cache_formal_artifacts import (
    INVALID_G14C_V4_RUN_ROOTS,
    checkpoint_freeze,
)
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    CellTransactionError,
    FormalCellLedger,
    atomic_write_json_create_only,
    stable_cell_id,
    stable_episode_id,
    validate_cell_ledger,
)
from src.evaluators.formal_phase_transaction import (
    PhaseCommandResult,
    PhaseTransactionError,
    TransactionalPhaseRunner,
    clock_measurement,
    validate_phase_ledger_v3,
)
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    READY_V6_VERDICT,
    expand_command_plan,
    expand_command_template,
    readiness_v6,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import (
    canonical_sha256,
    semantic_projection,
)
from src.runtime.formal_execution_environment import (
    ExecutionEnvironmentError,
    assert_child_environment_parity,
    probe_python_environment,
    resolve_execution_environment,
    scientific_environment_identity,
    validate_import_origins,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_4_20260825"
    / "protocol_v1_4_manifest.json"
)
V13_PATH = (
    ROOT
    / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
    / "protocol_v1_3_manifest.json"
)
RUN_A = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260824_110016_g14c_v4"
)
RUN_B = (
    ROOT
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260824_235839_g14c_v4"
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v13() -> dict:
    return json.loads(V13_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def environment_probe() -> dict:
    return probe_python_environment(sys.executable, clean_worktree_root=ROOT)


def identity(matrix: str = "matrix") -> CellExecutionIdentity:
    return CellExecutionIdentity(
        run_id="run",
        execution_commit="commit",
        protocol_semantic_sha256="protocol",
        resource_registry_semantic_sha256="resources",
        environment_fingerprint="environment",
        split_semantic_sha256="split",
        window_contract_semantic_sha256="windows",
        catalog_fingerprint="catalog",
        runtime_identity="runtime",
        command_matrix_sha256=matrix,
    )


class FakeTime:
    def __init__(self) -> None:
        self.utc_value = datetime(2026, 8, 25, tzinfo=timezone.utc)
        self.ns_value = 10_000_000_000

    def utc(self) -> datetime:
        return self.utc_value

    def ns(self) -> int:
        return self.ns_value

    def advance(self, seconds: float, *, utc_seconds: float | None = None) -> None:
        self.ns_value += int(seconds * 1e9)
        self.utc_value += timedelta(
            seconds=seconds if utc_seconds is None else utc_seconds
        )


def phase_runner(tmp_path: Path, clock: FakeTime) -> TransactionalPhaseRunner:
    return TransactionalPhaseRunner(
        output_root=tmp_path / "phase",
        run_identity_fingerprint="run",
        phase_order=["train"],
        utc_clock=clock.utc,
        monotonic_ns=clock.ns,
    )


def output_executor(runner: TransactionalPhaseRunner, clock: FakeTime, seconds: float = 1):
    def execute(_command):
        clock.advance(seconds)
        (runner.output_root / "output.json").write_text("{}", encoding="utf-8")
        return PhaseCommandResult(0)

    return execute


# Execution environment 1-12
def test_01_logical_environment_identity_excludes_absolute_python(environment_probe: dict) -> None:
    first = scientific_environment_identity(
        environment_probe, execution_commit="commit", source_tree_sha256="tree"
    )
    changed = dict(environment_probe)
    changed["sys_executable"] = "/relocated/python"
    second = scientific_environment_identity(
        changed, execution_commit="commit", source_tree_sha256="tree"
    )
    assert first == second
    assert "/Users/" not in json.dumps(first)


def test_02_absolute_python_resolution(protocol: dict) -> None:
    contract = protocol["formal_execution_environment_contract"]
    result = resolve_execution_environment(
        clean_worktree_root=ROOT,
        execution_commit=contract["scientific_identity"]["execution_commit"],
        python_executable=sys.executable,
        expected_identity=contract["scientific_identity"],
    )
    assert Path(result.python_executable).is_absolute()
    assert result.runtime_audit["resolution_source"] == "explicit_python_executable"


def test_03_clean_root_need_not_contain_dot_venv(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    shutil.copytree(ROOT / "src", root / "src")
    assert not (root / ".venv").exists()
    probe = probe_python_environment(
        sys.executable, clean_worktree_root=root, import_modules=("src",)
    )
    assert Path(probe["imports"]["src"]["file"]).is_relative_to(root)


def test_04_missing_interpreter_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ExecutionEnvironmentError, match="does not exist"):
        probe_python_environment(
            tmp_path / "missing-python", clean_worktree_root=ROOT
        )


def test_05_version_mismatch_is_rejected(protocol: dict) -> None:
    expected = deepcopy(protocol["formal_execution_environment_contract"]["scientific_identity"])
    expected["python_version"] = "0.0.0"
    with pytest.raises(ExecutionEnvironmentError, match="python_version"):
        resolve_execution_environment(
            clean_worktree_root=ROOT,
            execution_commit=expected["execution_commit"],
            python_executable=sys.executable,
            expected_identity=expected,
        )


def test_06_dependency_mismatch_is_rejected(protocol: dict) -> None:
    expected = deepcopy(protocol["formal_execution_environment_contract"]["scientific_identity"])
    expected["dependency_fingerprint"] = "drift"
    expected.pop("environment_fingerprint")
    with pytest.raises(ExecutionEnvironmentError, match="dependency_fingerprint"):
        resolve_execution_environment(
            clean_worktree_root=ROOT,
            execution_commit=expected["execution_commit"],
            python_executable=sys.executable,
            expected_identity=expected,
        )


def test_07_clean_import_origin(environment_probe: dict) -> None:
    assert validate_import_origins(
        environment_probe, clean_worktree_root=ROOT
    )["status"] == "pass"


def test_08_dirty_import_origin_is_rejected(environment_probe: dict, tmp_path: Path) -> None:
    fake = deepcopy(environment_probe)
    fake["imports"]["src"]["file"] = str(ROOT / "src/__init__.py")
    with pytest.raises(ExecutionEnvironmentError, match="clean worktree"):
        validate_import_origins(fake, clean_worktree_root=tmp_path)


def test_09_child_environment_parity(protocol: dict) -> None:
    contract = protocol["formal_execution_environment_contract"]
    result = resolve_execution_environment(
        clean_worktree_root=ROOT,
        execution_commit=contract["scientific_identity"]["execution_commit"],
        python_executable=sys.executable,
        expected_identity=contract["scientific_identity"],
    )
    assert assert_child_environment_parity(
        result,
        clean_worktree_root=ROOT,
        execution_commit=contract["scientific_identity"]["execution_commit"],
    )["status"] == "pass"


def test_10_relocation_does_not_change_semantic_hash(environment_probe: dict) -> None:
    identity_a = scientific_environment_identity(
        environment_probe, execution_commit="c", source_tree_sha256="t"
    )
    relocated = deepcopy(environment_probe)
    relocated["sys_prefix"] = "/another/venv"
    relocated["site_packages"] = ["/another/site-packages"]
    identity_b = scientific_environment_identity(
        relocated, execution_commit="c", source_tree_sha256="t"
    )
    assert identity_a["environment_fingerprint"] == identity_b["environment_fingerprint"]


def test_11_command_expansion_uses_absolute_python(protocol: dict) -> None:
    template = protocol["execution_contract"]["command_templates"]["preflight"]["argv"]
    values = dict(protocol["execution_contract"]["default_expansion_context"])
    values.update(python_executable=sys.executable, clean_worktree_root=str(ROOT))
    command = expand_command_template(template, values)
    assert command[0] == sys.executable and Path(command[0]).is_absolute()


def test_12_no_relative_or_dot_venv_python(protocol: dict) -> None:
    templates = protocol["execution_contract"]["command_templates"]
    assert all(spec["argv"][0] == "{python_executable}" for spec in templates.values())
    assert ".venv/bin/python" not in json.dumps(templates)


# Timing and phase transaction 13-21
def test_13_monotonic_duration_is_authoritative() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    measured = clock_measurement(
        started_at_utc=start,
        completed_at_utc=start + timedelta(seconds=20),
        monotonic_started_ns=1,
        monotonic_completed_ns=10_000_000_001,
        child_wall_clock_seconds=8,
    )
    assert measured["wall_clock_seconds"] == 10
    assert measured["wall_clock_adjustment_seconds"] == 10


@pytest.mark.parametrize(
    ("utc_delta", "status"),
    [(3700, "forward_system_clock_adjustment"), (-100, "backward_system_clock_adjustment")],
)
def test_14_clock_jumps_are_audited_not_rejected(utc_delta: int, status: str) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    measured = clock_measurement(
        started_at_utc=start,
        completed_at_utc=start + timedelta(seconds=utc_delta),
        monotonic_started_ns=0,
        monotonic_completed_ns=100_000_000_000,
        child_wall_clock_seconds=99,
    )
    assert measured["clock_consistency_status"] == status


def test_15_logical_five_hour_phase_is_valid() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    measured = clock_measurement(
        started_at_utc=start,
        completed_at_utc=start + timedelta(hours=5, seconds=120),
        monotonic_started_ns=0,
        monotonic_completed_ns=18_000_000_000_000,
        child_wall_clock_seconds=17_990,
    )
    assert measured["wall_clock_seconds"] == 18_000


def test_16_timezone_and_rounding_do_not_replace_monotonic() -> None:
    china = timezone(timedelta(hours=8))
    start = datetime(2026, 8, 25, 1, tzinfo=china)
    measured = clock_measurement(
        started_at_utc=start,
        completed_at_utc=start + timedelta(seconds=1.23456789),
        monotonic_started_ns=100,
        monotonic_completed_ns=1_234_567_990,
        child_wall_clock_seconds=1.0,
    )
    assert measured["wall_clock_seconds"] == pytest.approx(1.23456789)


def test_17_completion_candidate_persists_before_terminal(tmp_path: Path) -> None:
    clock = FakeTime()
    runner = phase_runner(tmp_path, clock)
    result = runner.run_phase(
        "train",
        commands=[["ok"]],
        input_hash="input",
        expected_outputs=["output.json"],
        executor=output_executor(runner, clock),
        stop_after_completion_candidate=True,
    )
    assert result["status"] == "completion_candidate_created"
    assert [item["status"] for item in runner.records()] == ["running", "completion_candidate"]


def test_18_finalize_only_and_idempotence(tmp_path: Path) -> None:
    clock = FakeTime()
    runner = phase_runner(tmp_path, clock)
    with pytest.raises(PhaseTransactionError, match="simulated"):
        runner.run_phase(
            "train",
            commands=[["ok"]],
            input_hash="input",
            expected_outputs=["output.json"],
            executor=output_executor(runner, clock),
            fail_terminal_append=True,
        )
    resumed = TransactionalPhaseRunner(
        output_root=runner.output_root,
        run_identity_fingerprint="run",
        phase_order=["train"],
        resume=True,
    )
    assert resumed.finalize_phase_only(
        "train", commands=[["ok"]], input_hash="input", expected_outputs=["output.json"]
    )["status"] == "completed"
    assert resumed.finalize_phase_only(
        "train", commands=[["ok"]], input_hash="input", expected_outputs=["output.json"]
    )["status"] == "already_finalized"


def test_19_finalize_output_drift_is_rejected(tmp_path: Path) -> None:
    clock = FakeTime()
    runner = phase_runner(tmp_path, clock)
    runner.run_phase(
        "train", commands=[["ok"]], input_hash="i", expected_outputs=["output.json"],
        executor=output_executor(runner, clock), stop_after_completion_candidate=True,
    )
    (runner.output_root / "output.json").write_text('{"drift": true}', encoding="utf-8")
    with pytest.raises(PhaseTransactionError, match="output drift"):
        runner.finalize_phase_only(
            "train", commands=[["ok"]], input_hash="i", expected_outputs=["output.json"]
        )


def test_20_terminal_is_immutable(tmp_path: Path) -> None:
    clock = FakeTime()
    runner = phase_runner(tmp_path, clock)
    runner.run_phase(
        "train", commands=[["ok"]], input_hash="i", expected_outputs=["output.json"],
        executor=output_executor(runner, clock),
    )
    with pytest.raises(PhaseTransactionError, match="input drift"):
        runner.run_phase(
            "train", commands=[["ok"]], input_hash="changed", expected_outputs=["output.json"]
        )


def test_21_phase_hash_chain_detects_tamper(tmp_path: Path) -> None:
    clock = FakeTime()
    runner = phase_runner(tmp_path, clock)
    runner.run_phase(
        "train", commands=[["ok"]], input_hash="i", expected_outputs=["output.json"],
        executor=output_executor(runner, clock),
    )
    records = runner.records()
    records[-1]["previous_record_hash"] = "tampered"
    with pytest.raises(PhaseTransactionError, match="previous hash"):
        validate_phase_ledger_v3(records)


# Cell and resume 22-33
def test_22_stable_cell_id() -> None:
    assert stable_cell_id("train", {"agent": "ppo", "seed": 7}) == stable_cell_id(
        "train", {"agent": "ppo", "seed": 7}
    )


def begin_with_artifact(ledger: FormalCellLedger, coordinates: dict, content: str = "{}"):
    begun = ledger.begin_cell(
        phase="train", coordinates=coordinates, command=["run"], input_hash="input"
    )
    staging = Path(begun["record"]["staging_path"])
    (staging / "summary.json").write_text(content, encoding="utf-8")
    return begun


def test_23_staging_and_atomic_commit(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = begin_with_artifact(ledger, {"seed": 7})
    event = ledger.commit_cell(begun["cell_id"], required_paths=["summary.json"])
    assert event["status"] == "committed"
    assert (Path(event["committed_path"]) / "committed_marker.json").is_file()


def test_24_partial_attempt_is_preserved_and_new_attempt_started(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ledger = FormalCellLedger(run_root=root, identity=identity())
    first = ledger.begin_cell(
        phase="train", coordinates={"seed": 7}, command=["run"], input_hash="input"
    )
    first_path = Path(first["record"]["staging_path"])
    (first_path / "partial.txt").write_text("partial", encoding="utf-8")
    resumed = FormalCellLedger(run_root=root, identity=identity(), resume=True)
    second = resumed.begin_cell(
        phase="train", coordinates={"seed": 7}, command=["run"], input_hash="input"
    )
    assert second["record"]["attempt"] == 2 and first_path.is_dir()
    assert any(item["status"] == "incomplete" for item in resumed.records())


def test_25_committed_cell_is_skipped_without_rerun(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = begin_with_artifact(ledger, {"seed": 7})
    ledger.commit_cell(begun["cell_id"])
    assert ledger.begin_cell(
        phase="train", coordinates={"seed": 7}, command=["run"], input_hash="input"
    )["status"] == "skipped_committed"


def test_26_incomplete_cell_restarts_from_cell_start(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    first = ledger.begin_cell(
        phase="train", coordinates={"cell": 1}, command=["run"], input_hash="input"
    )
    resumed = FormalCellLedger(run_root=ledger.run_root, identity=identity(), resume=True)
    second = resumed.begin_cell(
        phase="train", coordinates={"cell": 1}, command=["run"], input_hash="input"
    )
    assert first["record"]["attempt"] == 1 and second["record"]["attempt"] == 2


def test_27_terminal_cell_failure_blocks_resume(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = ledger.begin_cell(
        phase="train", coordinates={"cell": 1}, command=["run"], input_hash="input"
    )
    ledger.fail_cell(
        begun["cell_id"], return_code=1, classification="failure", retryable=False
    )
    with pytest.raises(CellTransactionError, match="terminal"):
        ledger.begin_cell(
            phase="train", coordinates={"cell": 1}, command=["run"], input_hash="input"
        )


def test_28_cross_run_identity_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    FormalCellLedger(run_root=root, identity=identity("a"))
    with pytest.raises(CellTransactionError, match="drift"):
        FormalCellLedger(run_root=root, identity=identity("b"), resume=True)


def test_29_protocol_or_environment_drift_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    original = identity()
    FormalCellLedger(run_root=root, identity=original)
    drift = CellExecutionIdentity(**{**original.to_dict(), "environment_fingerprint": "drift"})
    with pytest.raises(CellTransactionError, match="drift"):
        FormalCellLedger(run_root=root, identity=drift, resume=True)


def test_30_interruption_at_75_of_150_and_resume(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ledger = FormalCellLedger(run_root=root, identity=identity("150"))
    expected = []
    for index in range(150):
        cell_id = stable_cell_id("train", {"index": index})
        expected.append(cell_id)
        if index >= 75:
            continue
        begun = begin_with_artifact(ledger, {"index": index})
        ledger.commit_cell(begun["cell_id"])
    resumed = FormalCellLedger(run_root=root, identity=identity("150"), resume=True)
    skipped = 0
    for index in range(150):
        result = resumed.begin_cell(
            phase="train", coordinates={"index": index}, command=["run"], input_hash="input"
        )
        if result["status"] == "skipped_committed":
            skipped += 1
            continue
        staging = Path(result["record"]["staging_path"])
        (staging / "summary.json").write_text("{}", encoding="utf-8")
        resumed.commit_cell(result["cell_id"])
    assert skipped == 75
    assert resumed.assert_complete_matrix(phase="train", expected_cell_ids=expected)[
        "committed_cell_count"
    ] == 150


def test_31_no_duplicate_cells() -> None:
    ids = [stable_cell_id("train", {"index": index}) for index in range(150)]
    assert len(set(ids)) == 150


def test_32_no_duplicate_episode_ids() -> None:
    ids = [stable_episode_id({"window": i, "seed": 7}) for i in range(12)]
    assert len(set(ids)) == 12


def test_33_aggregate_reads_committed_only(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    committed = begin_with_artifact(ledger, {"cell": 1})
    ledger.commit_cell(committed["cell_id"])
    ledger.begin_cell(
        phase="train", coordinates={"cell": 2}, command=["run"], input_hash="input"
    )
    assert len(ledger.committed_records(phase="train")) == 1


# Dev/formal transaction 34-39
def test_34_candidate_resume_skips_committed(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = ledger.begin_cell(
        phase="dev_select", coordinates={"candidate": "a"}, command=["eval"], input_hash="i"
    )
    staging = Path(begun["record"]["staging_path"])
    (staging / "candidate.json").write_text("{}", encoding="utf-8")
    ledger.commit_cell(begun["cell_id"])
    assert ledger.begin_cell(
        phase="dev_select", coordinates={"candidate": "a"}, command=["eval"], input_hash="i"
    )["status"] == "skipped_committed"


def test_35_partial_candidate_set_cannot_select(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = ledger.begin_cell(
        phase="dev_select", coordinates={"candidate": 1}, command=["eval"], input_hash="i"
    )
    staging = Path(begun["record"]["staging_path"])
    (staging / "candidate.json").write_text("{}", encoding="utf-8")
    ledger.commit_cell(begun["cell_id"])
    expected = [stable_cell_id("dev_select", {"candidate": i}) for i in (1, 2)]
    with pytest.raises(CellTransactionError, match="partial"):
        ledger.assert_complete_matrix(phase="dev_select", expected_cell_ids=expected)


def test_36_selection_manifest_is_atomic_and_create_only(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    atomic_write_json_create_only(path, {"selected": [1]})
    assert json.loads(path.read_text())["selected"] == [1]
    with pytest.raises(CellTransactionError, match="already exists"):
        atomic_write_json_create_only(path, {"selected": [2]})


def test_37_formal_cell_resume(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = ledger.begin_cell(
        phase="formal_controller", coordinates={"episode": 1}, command=["eval"], input_hash="i"
    )
    staging = Path(begun["record"]["staging_path"])
    for name in ("summary.json", "row.json", "events.json", "audit.json"):
        (staging / name).write_text("{}", encoding="utf-8")
    ledger.commit_cell(
        begun["cell_id"],
        required_paths=["summary.json", "row.json", "events.json", "audit.json"],
    )
    assert ledger.begin_cell(
        phase="formal_controller", coordinates={"episode": 1}, command=["eval"], input_hash="i"
    )["status"] == "skipped_committed"


def test_38_raw_row_event_audit_commit_together(tmp_path: Path) -> None:
    ledger = FormalCellLedger(run_root=tmp_path / "run", identity=identity())
    begun = ledger.begin_cell(
        phase="formal_controller", coordinates={"episode": 1}, command=["eval"], input_hash="i"
    )
    staging = Path(begun["record"]["staging_path"])
    for name in ("summary.json", "row.json", "events.json"):
        (staging / name).write_text("{}", encoding="utf-8")
    with pytest.raises(CellTransactionError, match="missing"):
        ledger.commit_cell(
            begun["cell_id"],
            required_paths=["summary.json", "row.json", "events.json", "audit.json"],
        )


def test_39_statistics_outer_count_does_not_expand_on_resume() -> None:
    before = {stable_episode_id({"window": i}) for i in range(12)}
    after = before | {stable_episode_id({"window": i}) for i in range(12)}
    assert len(before) == len(after) == 12


# Protocol 40-46
def test_40_two_v4_invalid_references_are_frozen(protocol: dict) -> None:
    refs = protocol["supersession"]["invalid_g14c_v4_runs"]
    assert {item["run_id"] for item in refs} == {RUN_A.name, RUN_B.name}
    assert all(not item["resume_allowed"] for item in refs)


def test_41_protocol_v14_hash_and_validation(protocol: dict) -> None:
    report = validate_protocol_v1_1(protocol)
    assert report["protocol_version"] == "1.4.0"
    assert protocol["hashes"]["semantic_sha256"] == canonical_sha256(
        semantic_projection({key: value for key, value in protocol.items() if key != "hashes"})
    )


def test_42_scientific_fields_are_unchanged(protocol: dict, v13: dict) -> None:
    keys = (
        "workload", "agent_matrix", "seed_plan", "training_budget",
        "typed_catalog_and_capacity", "endpoints", "ablation_and_support",
        "statistics", "claim_evidence_map", "comparisons",
    )
    assert all(protocol[key] == v13[key] for key in keys)
    assert protocol["identity"]["split_semantic_sha256"] == v13["identity"]["split_semantic_sha256"]


def test_43_holdout_remains_unavailable(protocol: dict) -> None:
    holdout = protocol["holdout_execution_contract"]
    assert holdout["sealed"] and not holdout["opened"] and not holdout["consumed_permanently"]
    assert protocol["execution_contract"]["holdout_capability"] is False


def test_44_readiness_failure_and_success() -> None:
    names = {
        "two_v4_failures_registered", "clean_worktree_without_local_venv",
        "all_commands_use_resolved_interpreter", "clean_import_origin",
        "environment_fingerprint", "long_phase_and_clock_jump",
        "phase_transaction_and_finalize_only", "cell_ledger_and_atomic_commit",
        "same_run_resume", "interruption_75_of_150", "dev_formal_committed_only",
        "old_runs_hard_rejected", "holdout_sealed", "no_formal_performance_results",
    }
    assert readiness_v6({name: True for name in names}) == READY_V6_VERDICT
    checks = {name: True for name in names}
    checks["same_run_resume"] = False
    assert readiness_v6(checks) == "BLOCKED_G14R4_READINESS_V6"


def test_45_json_round_trip(protocol: dict) -> None:
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_46_nan_inf_rejected(protocol: dict, bad: float) -> None:
    changed = deepcopy(protocol)
    changed["training_budget"]["batch_size"] = bad
    with pytest.raises(FormalExecutionError, match="non-finite"):
        validate_protocol_v1_1(changed)


def test_47_matrix_absolute_sentinel_is_resolved(protocol: dict) -> None:
    context = dict(protocol["execution_contract"]["default_expansion_context"])
    context.update(
        output_root="/tmp/formal-run",
        python_executable=sys.executable,
        clean_worktree_root=str(ROOT),
    )
    plans = [
        expand_command_plan(spec, context)
        for spec in protocol["execution_contract"]["command_templates"].values()
    ]
    assert not any(
        "/ABSOLUTE/" in token
        for plan in plans
        for command in plan["commands"]
        for token in command
    )


def test_48_retryable_command_runs_exactly_once_more(tmp_path: Path) -> None:
    clock = FakeTime()
    runner = phase_runner(tmp_path, clock)
    calls = 0

    def execute(_command):
        nonlocal calls
        calls += 1
        clock.advance(1)
        if calls == 1:
            return PhaseCommandResult(75, "", "retry")
        (runner.output_root / "output.json").write_text("{}", encoding="utf-8")
        return PhaseCommandResult(0)

    result = runner.run_phase(
        "train",
        commands=[["retryable"]],
        input_hash="input",
        expected_outputs=["output.json"],
        executor=execute,
        infrastructure_retries=1,
    )
    assert result["status"] == "completed" and calls == 2


def test_49_publish_before_ledger_append_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    ledger = FormalCellLedger(run_root=root, identity=identity())
    begun = begin_with_artifact(ledger, {"seed": 7})
    original_append = ledger._append

    def fail_append(_record):
        raise RuntimeError("simulated post-publish interruption")

    monkeypatch.setattr(ledger, "_append", fail_append)
    with pytest.raises(RuntimeError, match="post-publish"):
        ledger.commit_cell(begun["cell_id"])
    monkeypatch.setattr(ledger, "_append", original_append)
    resumed = FormalCellLedger(run_root=root, identity=identity(), resume=True)
    result = resumed.begin_cell(
        phase="train", coordinates={"seed": 7}, command=["run"], input_hash="input"
    )
    assert result["status"] == "skipped_committed"


@pytest.mark.parametrize("invalid_root", INVALID_G14C_V4_RUN_ROOTS)
def test_50_invalid_v4_checkpoint_root_is_hard_rejected(
    protocol: dict, tmp_path: Path, invalid_root: Path
) -> None:
    selection = {
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "selection_sha256": "selection",
        "selected": [
            {
                "agent_name": "ppo",
                "seed": 7,
                "capacity_label": "medium_576mb",
                "update_index": 4,
                "checkpoint_path": str(invalid_root / "candidate.pt"),
                "checkpoint_sha256": "0" * 64,
            }
        ],
    }
    (tmp_path / "dev_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="G14C v3/v4"):
        checkpoint_freeze(tmp_path, protocol)


def test_51_phase_absolute_sanity_bound_is_enforced() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(PhaseTransactionError, match="absolute sanity"):
        clock_measurement(
            started_at_utc=start,
            completed_at_utc=start + timedelta(days=4),
            monotonic_started_ns=0,
            monotonic_completed_ns=4 * 24 * 3600 * 1_000_000_000,
            child_wall_clock_seconds=1,
        )
