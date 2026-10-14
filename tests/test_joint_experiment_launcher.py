from argparse import Namespace
import json
import sys

import pytest

from scripts.launch_eight_vehicle_joint_experiment import _agent_options, _episode
from scripts import launch_eight_vehicle_joint_experiment as launcher
from src.agents.registry import build_agent
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState, WorkflowNode


def _fixture():
    frames = [{"time_index": i, "source_segment_id": "synthetic",
               "vehicles": [VehicleState("v0", 0, 0, 1, "base"),
                            VehicleState("v1", 8, 0, 1, "base")]} for i in range(6)]
    workflows = []
    for i in range(2):
        node = WorkflowNode(f"n{i}", "node", "base", "adapter", 1024, 10)
        workflows.append(WorkflowGraphState(f"w{i}", [node], [], [node.node_id],
                                            current_node_id=node.node_id))
    return ({"frames": frames, "vehicle_ids": ["v0", "v1"], "frame_offset": 0,
             "time_start": 0, "time_end": 5}, workflows,
            [RSUState("a", 0, 0, 100), RSUState("b", 10, 0, 100)])


def test_rollout_retains_actual_policy_metadata_after_serialization():
    class Agent:
        def act(self, observation, info):
            assert info["run_metadata"]["policy_evaluation_mode"] == "raw_policy"
            return 3, {"log_prob": -0.25, "value": 0.0}
    summary, rows, events = _episode(Agent(), *_fixture(), deterministic=False, collect=True)
    serialized = json.loads(json.dumps(rows))
    assert len(serialized) == 12 and len(events) == 12
    assert summary["action_selection"] == "sample"
    assert all(row["decision_info"]["run_metadata"]["policy_evaluation_mode"] == "raw_policy"
               for row in serialized)
    assert all(row["decision_info"]["deterministic_policy"] is False for row in serialized)


@pytest.mark.parametrize("name", ["sa_ghmappo", "mappo", "ppo", "dqn", "dag_offload_drl",
                                  "cache_offload_drl", "dt_handoff_drl"])
def test_v2_registry_rollout_learn_and_checkpoint_roundtrip(name, tmp_path):
    args = Namespace(environment_revision="v2", sa_auxiliary_coef=0.0)
    options = _agent_options(name, args)
    if name == "dqn":
        options["min_replay_size"] = 4
    agent = build_agent(name, random_seed=13, train_epochs=1, batch_size=4, **options)
    summary, rows, events = _episode(agent, *_fixture(), deterministic=False, collect=True,
                                   environment_options={"environment_revision": "v2"})
    assert rows and events
    assert all("centralized_critic_state" in row["decision_info"] for row in rows)
    update = agent.learn(rows)
    assert isinstance(update, dict)
    assert not update.get("policy_update_skipped", False)
    assert update["update_count"] == 1
    checkpoint = tmp_path / f"{name}.pt"
    agent.save(str(checkpoint))
    restored = build_agent(name, random_seed=13, train_epochs=1, batch_size=4, **options)
    restored.load(str(checkpoint))
    expected, _, _ = _episode(agent, *_fixture(), deterministic=True, collect=False,
                             environment_options={"environment_revision": "v2"})
    actual, _, _ = _episode(restored, *_fixture(), deterministic=True, collect=False,
                           environment_options={"environment_revision": "v2"})
    assert actual == expected


def test_finished_request_retains_team_reward_after_its_last_action():
    class Agent:
        def act(self, observation, info):
            vid = info["semantic_state"]["primary_vehicle_id"]
            return (2 if vid == "v0" else 3), {"log_prob": 0.0, "value": 0.0}
    _, rows, _ = _episode(Agent(), *_fixture(), deterministic=False, collect=True)
    early = [row for row in rows if row["decision_info"]["semantic_state"]["primary_vehicle_id"] == "v0"]
    late = [row for row in rows if row["decision_info"]["semantic_state"]["primary_vehicle_id"] == "v1"]
    assert len(early) == 4 and len(late) == 6
    assert early[-1]["return"] == pytest.approx(late[3]["return"])
    assert early[-1]["reward"] == pytest.approx(early[-1]["return"])
    assert early[-1]["team_terminal_continuation_reward"] < 0
    assert early[0]["return"] == pytest.approx(late[0]["return"])


def test_main_v2_writes_complete_development_package_without_formal(monkeypatch, tmp_path):
    window, workflows, rsus = _fixture()
    monkeypatch.setattr(launcher, "_load_protocol", lambda args: (
        {name: [window] for name in ("train", "dev", "formal", "holdout")}, workflows, rsus))
    raw = tmp_path / "fixture.csv"
    raw.write_text("synthetic_fixture_only")
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["launch_eight_vehicle_joint_experiment.py",
        "--environment_revision", "v2", "--smoke", "--episodes", "1", "--update_every", "1",
        "--agents", "sa_ghmappo", "mappo", "--seeds", "13",
        "--mobility_csv_path", str(raw), "--workflow_csv_path", str(raw), "--output_root", str(output)])
    launcher.main()
    run = next(output.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["protocol"] == "eight_vehicle_joint_sim_v2_development"
    assert manifest["checkpoint_count"] == 2
    assert manifest["costs"]["cost_calibration"] == "controlled_synthetic_not_measured"
    assert "mechanism_qualification" in manifest
    assert not (run / "formal.jsonl").exists() and not (run / "holdout.jsonl").exists()
    assert len((run / "dev_sampled.jsonl").read_text().splitlines()) == 2
    events = [json.loads(line) for line in (run / "dev_events.jsonl").read_text().splitlines()]
    assert {event["action_selection"] for event in events} == {"greedy", "sample"}
    assert any("frame_cost_receipt" in event for event in events)


def test_main_v2_cold_mask_records_distinct_contract(monkeypatch, tmp_path):
    window, workflows, rsus = _fixture()
    monkeypatch.setattr(launcher, "_load_protocol", lambda args: (
        {name: [window] for name in ("train", "dev", "formal", "holdout")}, workflows, rsus))
    raw = tmp_path / "fixture.csv"
    raw.write_text("synthetic_fixture_only")
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["launch_eight_vehicle_joint_experiment.py",
        "--environment_revision", "v2", "--smoke", "--strict_cold_cache_mask",
        "--episodes", "1", "--update_every", "1", "--agents", "mappo", "--seeds", "13",
        "--mobility_csv_path", str(raw), "--workflow_csv_path", str(raw),
        "--output_root", str(output)])
    launcher.main()
    manifest = json.loads((next(output.iterdir()) / "manifest.json").read_text())
    assert manifest["protocol"] == "eight_vehicle_joint_sim_v2_cold_mask_development"
    assert manifest["action_contract"] == "semantic_discrete_5_per_request_v2_cold_cache_mask"
    assert manifest["environment_options"]["strict_cold_cache_mask"] is True
    assert manifest["costs"]["strict_cold_cache_mask"] is True


@pytest.mark.parametrize("name", ["sa_ghmappo", "mappo", "cache_offload_drl"])
def test_development_curve_saves_restorable_snapshots_without_opening_formal(monkeypatch, tmp_path, name):
    window, workflows, rsus = _fixture()
    monkeypatch.setattr(launcher, "_load_protocol", lambda args: (
        {name: [window] for name in ("train", "dev", "formal", "holdout")}, workflows, rsus))
    raw = tmp_path / "fixture.csv"
    raw.write_text("synthetic_fixture_only")
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["launch_eight_vehicle_joint_experiment.py",
        "--environment_revision", "v2", "--development_only",
        "--episodes", "4", "--update_every", "2", "--development_eval_every", "2",
        "--agents", name, "--seeds", "13", "--mobility_csv_path", str(raw),
        "--workflow_csv_path", str(raw), "--output_root", str(output)])
    launcher.main()
    run = next(output.iterdir())
    curve = [json.loads(line) for line in (run / "dev_curve.jsonl").read_text().splitlines()]
    checkpoints = [json.loads(line) for line in (run / "curve_checkpoints.jsonl").read_text().splitlines()]
    final = json.loads((run / "dev.jsonl").read_text().strip())
    assert [row["episode"] for row in curve] == [2, 4]
    assert [row["episode"] for row in checkpoints] == [2, 4]
    assert all(launcher._hash_file(launcher.Path(row["path"])) == row["sha256"]
               for row in checkpoints)
    assert final["reward"] == curve[-1]["reward"]
    assert not (run / "formal.jsonl").exists() and not (run / "holdout.jsonl").exists()
    control_output = tmp_path / "control_runs"
    monkeypatch.setattr(sys, "argv", ["launch_eight_vehicle_joint_experiment.py",
        "--environment_revision", "v2", "--development_only",
        "--episodes", "4", "--update_every", "2", "--agents", name, "--seeds", "13",
        "--mobility_csv_path", str(raw), "--workflow_csv_path", str(raw),
        "--output_root", str(control_output)])
    launcher.main()
    control = next(control_output.iterdir())
    assert (run / "train.jsonl").read_text() == (control / "train.jsonl").read_text()
    assert (run / "updates.jsonl").read_text() == (control / "updates.jsonl").read_text()


def test_v2_formal_execution_rejected_before_loading_data(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher", "--environment_revision", "v2"])
    monkeypatch.setattr(launcher, "_load_protocol", lambda args: pytest.fail("formal must not load"))
    with pytest.raises(ValueError, match="development-only"):
        launcher.main()
