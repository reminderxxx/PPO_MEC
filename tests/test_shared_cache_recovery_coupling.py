from __future__ import annotations

import json
from pathlib import Path

from scripts.run_shared_cache_recovery_coupling import build_env, run_branch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment/shared_cache_recovery_coupling_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _instance(config: dict, prefix: str) -> dict:
    return next(row for row in config["design_points"] if row["instance_id"].startswith(prefix))


def _later_transfer_bytes(branch: dict) -> int:
    return sum(branch["steps"][1]["transfer_bytes_by_type"].values())


def test_frozen_a_b_c_points_have_the_declared_native_coupling(tmp_path: Path) -> None:
    config = _config()
    for prefix, expected_change in (("A_", False), ("B_", False), ("C_", True)):
        instance = _instance(config, prefix)
        no_prepare = run_branch(instance, config, tmp_path / prefix / "no_prepare", 0)
        prepare = run_branch(instance, config, tmp_path / prefix / "prepare", 4)
        assert no_prepare["feasible"] and prepare["feasible"]
        assert no_prepare["summary"]["completed_node_count"] == 2
        assert prepare["summary"]["completed_node_count"] == 2
        assert (_later_transfer_bytes(no_prepare) != _later_transfer_bytes(prepare)) is expected_change


def test_reset_restores_template_and_breaks_cross_workflow_residency(tmp_path: Path) -> None:
    config = _config()
    instance = _instance(config, "C_")
    env = build_env(instance, config, tmp_path / "packages")
    _, reset = env.reset(seed=int(config["seed"]))
    initial = reset["cache_trace_snapshot"]
    initial_target = next(row for row in initial["rsus"] if row["rsu_id"] == "rsu_b")
    assert [row["object_id"] for row in initial_target["residents"]] == ["base:b1", "adapter:b1.a0"]

    _, _, _, _, first = env.step(4)
    changed_target = next(
        row for row in first["cache_trace_snapshot"]["rsus"] if row["rsu_id"] == "rsu_b"
    )
    assert [row["object_id"] for row in changed_target["residents"]] == [
        "base:b0",
        "adapter:b0.a0",
    ]

    _, reset_again = env.reset(seed=int(config["seed"]))
    restored_target = next(
        row for row in reset_again["cache_trace_snapshot"]["rsus"] if row["rsu_id"] == "rsu_b"
    )
    assert [row["object_id"] for row in restored_target["residents"]] == [
        "base:b1",
        "adapter:b1.a0",
    ]
