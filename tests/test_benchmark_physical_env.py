"""Non-formal physical benchmark identity and output contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_physical_env as benchmark
from src.metrics.physical_event_metrics import PhysicalEvent
from src.runtime.physical_env_config import physical_env_config_identity


CONFIG = {
    "schema_version": "physical_vec_env_v1",
    "parameter_evidence": "synthetic",
    "time_unit": "s",
    "size_unit": "byte",
    "control_step_s": 1.0,
    "backhaul_bytes_per_s": 10_000_000.0,
    "wireless_bytes_per_s": 5_000_000.0,
    "compute_rate_work_units_per_s": 100.0,
    "compute_slots_per_rsu": 1,
    "cache_capacity_bytes_per_rsu": 320 * 1024 * 1024,
}


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path, *, agents: list[str] | None = None):
    config_path = _write(tmp_path / "physical.json", CONFIG)
    plan_path = _write(tmp_path / "windows.json", {
        "mobility_source": "ngsim",
        "selected_window_plan": [
            {"window_id": "window_fixed", "frame_offset": 0, "window_length": 24,
             "window_class": "mechanism_activating"}
        ],
    })
    mobility = tmp_path / "ngsim.csv"
    workflow = tmp_path / "alibaba.csv"
    mobility.write_text("frame,vehicle\n1,v\n", encoding="utf-8")
    workflow.write_text("job,task\n1,1\n", encoding="utf-8")
    manifest_path = tmp_path / "checkpoints.json"
    if "ppo" in (agents or []):
        checkpoint_path = tmp_path / "ppo.pt"
        torch.save({"training_metadata": {
            "environment_version": "physical_vec_env_v1",
            "physical_environment_config_sha256": physical_env_config_identity(CONFIG),
            "agent_name": "ppo", "seed": 7, "run_id": "train-physical",
        }}, checkpoint_path)
        _write(manifest_path, {"ppo": {"7": {"path": "ppo.pt"}}})
    args = benchmark.build_parser().parse_args([
        "--physical_env_config", str(config_path),
        "--window_plan_path", str(plan_path),
        "--mobility_csv_path", str(mobility),
        "--workflow_csv_path", str(workflow),
        "--agents", *(agents or ["reactive_greedy"]),
        "--seeds", "7",
        "--output_root", str(tmp_path / "out"),
        *(["--checkpoint_manifest_path", str(manifest_path)] if manifest_path.exists() else []),
    ])
    return args, config_path, manifest_path


def _valid_trace() -> list[dict]:
    return [
        PhysicalEvent("e1", "request_created", 0, "r", workflow_id="wf").to_dict(),
        PhysicalEvent("e2", "delivery_started", 1, "r", workflow_id="wf", operation_id="d").to_dict(),
        PhysicalEvent("e3", "delivery_completed", 2, "r", workflow_id="wf", operation_id="d").to_dict(),
        PhysicalEvent("e4", "request_completed", 2, "r", workflow_id="wf").to_dict(),
    ]


def test_rejects_legacy_checkpoint_before_running(tmp_path: Path) -> None:
    args, _, manifest_path = _inputs(tmp_path, agents=["ppo"])
    checkpoint = tmp_path / "ppo.pt"
    torch.save({"training_metadata": {"agent_name": "ppo", "seed": 7}}, checkpoint)
    with pytest.raises(ValueError, match="environment_version mismatch"):
        benchmark.run_benchmark(args)
    assert not (tmp_path / "out").exists()
    assert manifest_path.exists()


def test_rejects_missing_config_and_checkpoint_manifest(tmp_path: Path) -> None:
    args, config_path, manifest_path = _inputs(tmp_path, agents=["ppo"])
    manifest_path.unlink()
    with pytest.raises(FileNotFoundError):
        benchmark.run_benchmark(args)
    args.checkpoint_manifest_path = None
    with pytest.raises(ValueError, match="missing physical checkpoint manifest entry"):
        benchmark.run_benchmark(args)
    config_path.unlink()
    with pytest.raises(FileNotFoundError, match="required input missing"):
        benchmark.run_benchmark(args)


def test_rejects_duplicate_window_ids(tmp_path: Path) -> None:
    plan = _write(tmp_path / "windows.json", {
        "selected_window_plan": [
            {"window_id": "same", "frame_offset": 0, "window_length": 24, "window_class": "a"},
            {"window_id": "same", "frame_offset": 24, "window_length": 24, "window_class": "b"},
        ]
    })
    with pytest.raises(ValueError, match="duplicate window_id"):
        benchmark.load_frozen_window_plan(plan)


def test_writes_per_episode_trace_metrics_identity_and_raw_rows(tmp_path: Path, monkeypatch) -> None:
    args, _, _ = _inputs(tmp_path, agents=["ppo", "reactive_greedy"])
    calls = []
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        benchmark, "build_selected_workflow_states",
        lambda **kwargs: [SimpleNamespace(workflow_id="wf")],
    )
    monkeypatch.setattr(
        benchmark, "load_window_bundle",
        lambda **kwargs: SimpleNamespace(rsu_metadata={"window_id": "window_fixed"}),
    )

    def fake_episode(**kwargs):
        calls.append(kwargs)
        return {
            "physical_event_trace": _valid_trace(),
            "workflow_physical_metrics": benchmark.reduce_physical_events(
                _valid_trace()
            ).to_dict()["requests"][0],
            "run_info": {"agent_name": kwargs["agent_name"]},
        }

    monkeypatch.setattr(benchmark, "run_real_episode", fake_episode)
    output = benchmark.run_benchmark(args)
    assert len(calls) == 2
    assert all(call["physical_env_config"]["control_step_s"] == 1.0 for call in calls)
    assert all(call["run_metadata"]["evidence_level"] == "non_formal_diagnostic" for call in calls)
    assert all(call["mobility_source"] == "ngsim" for call in calls)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["formal"] is False
    assert manifest["paper_claim_ready"] is False
    assert manifest["episode_count"] == 2
    rows = json.loads((output / "rows.json").read_text())
    assert len(rows) == 2
    assert all(row["physical_metrics_availability"] == "available" for row in rows)
    for row in rows:
        assert Path(row["event_trace_path"]).is_file()
        assert Path(row["physical_metrics_path"]).is_file()
    ppo_identity = json.loads((output / "episodes/window_fixed/wf/ppo/seed_7/identity.json").read_text())
    assert ppo_identity["checkpoint"]["environment_version"] == "physical_vec_env_v1"
    assert (output / "rows.csv").is_file()


def test_fails_on_window_identity_mismatch_before_episode(tmp_path: Path, monkeypatch) -> None:
    args, _, _ = _inputs(tmp_path)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(benchmark, "build_selected_workflow_states", lambda **kwargs: [SimpleNamespace(workflow_id="wf")])
    monkeypatch.setattr(benchmark, "load_window_bundle", lambda **kwargs: SimpleNamespace(rsu_metadata={"window_id": "other"}))
    with pytest.raises(ValueError, match="window identity mismatch"):
        benchmark.run_benchmark(args)
