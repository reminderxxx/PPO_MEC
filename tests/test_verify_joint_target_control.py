import hashlib
import json
import sys

import pytest

from scripts import verify_joint_target_control as runner
from tests.test_multi_request_target_control_env import make_env


def source_fixture(tmp_path, monkeypatch):
    env = make_env(count=4, nodes=3)
    window = {"frame_offset": 0, "time_start": 0, "time_end": 490,
              "vehicle_ids": sorted(env.workflows), "frames": env.frames}
    raw = tmp_path / "raw.csv"
    raw.write_text("synthetic fixture")
    public = {key: value for key, value in window.items() if key != "frames"}
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {"development_only": True, "environment_revision": "v2", "run_id": "synthetic_only",
        "command": ["launcher", "--mobility_csv_path", str(raw), "--workflow_csv_path", str(raw)],
        "raw_data_sha256": {"ngsim": digest, "alibaba": digest},
        "splits": {"train": [public], "dev": [public]}, "costs": env.configuration(),
        "environment_options": {"environment_revision": "v2", "workload_bytes_per_unit": 2}}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "_load_protocol", lambda _: (
        {name: [window] for name in ("train", "dev", "formal", "holdout")},
        list(env.workflow_templates.values()), env.rsus))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["runner", "--source_manifest", str(path),
                                      "--windows", "1", "--output_root", str(output)])
    return path, raw, output


def test_engineering_entry_verifies_source_and_produces_independent_rule_traces(tmp_path, monkeypatch):
    _, _, output = source_fixture(tmp_path, monkeypatch)
    runner.main()
    run = next(output.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["split"] == "train"
    assert manifest["evidence_scope"] == "engineering_rules_no_RL_no_formal_holdout"
    assert manifest["summary"]["reachable_balanced_reactive"]["mean_reward"] > manifest["summary"]["associated_reactive"]["mean_reward"]
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["configuration"]["environment_revision"] == "v3" for row in rows)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    for row in rows:
        receipts = [e["frame_cost_receipt"] for e in events if e["policy"] == row["policy"]
                    and "frame_cost_receipt" in e]
        assert len(receipts) == row["steps"]
        assert sum(sum(r["reward_components"].values()) for r in receipts) == pytest.approx(row["reward"])
    assert not (run / "formal.jsonl").exists() and not (run / "holdout.jsonl").exists()


@pytest.mark.parametrize("bad", ["data", "split"])
def test_source_change_fails_before_output_creation(tmp_path, monkeypatch, bad):
    path, raw, output = source_fixture(tmp_path, monkeypatch)
    if bad == "data":
        raw.write_text("changed input")
    else:
        manifest = json.loads(path.read_text())
        manifest["splits"]["train"][0]["frame_offset"] = 999
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash changed|split differs"):
        runner.main()
    assert not output.exists()
