import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.prepare_drivelm_pilot import (
    ARMS, CAMERAS, MAX_BYTES, STAGES, build_pilot_input, build_pilot_plan,
)


def sample(scenes=2):
    frame = {
        "QA": {s: [{"Q": f"QUESTION_{s}", "A": "SECRET_ANSWER"},
                   {"Q": "UNSELECTED_QUESTION", "A": "OTHER_SECRET"}] for s in STAGES},
        "key_object_infos": {"object": "SECRET_OBJECT"},
        "image_paths": {c: f"../untrusted/{c}/image.jpg" for c in CAMERAS},
    }
    return {f"scene_{i}": {"key_frames": {"z_frame": copy.deepcopy(frame),
                                           "a_frame": copy.deepcopy(frame)}} for i in range(scenes)}


def prepare(data=None):
    raw = json.dumps(sample() if data is None else data).encode()
    return raw, build_pilot_plan(raw, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("count,budget", [(1, 8), (2, 16)])
def test_matched_metadata_plan_and_budget(count, budget):
    raw, plan = prepare(sample(count))
    assert plan["planned_model_calls"] == len(plan["tasks"]) == budget
    assert plan["actual_model_calls"] == 0 and plan["model_call_cap"] == 16
    assert plan["runtime_eligible"] is False and plan["performance_evidence"] is False
    assert all(t["frame_id"] == "a_frame" for t in plan["tasks"])
    assert all(t["question_ref"]["index"] == 0 for t in plan["tasks"])
    assert all(t["image_camera_names"] == list(CAMERAS) for t in plan["tasks"])
    encoded = json.dumps(plan)
    assert not any(x in encoded for x in ("QUESTION_", "SECRET", "untrusted"))
    for scene in range(count):
        left = plan["tasks"][scene * 8:scene * 8 + 4]
        right = plan["tasks"][scene * 8 + 4:scene * 8 + 8]
        for a, b in zip(left, right):
            assert a["question_ref"] == b["question_ref"]
            assert (a["scene_id"], a["frame_id"]) == (b["scene_id"], b["frame_id"])
        outputs = {}
        for t in right:
            payload = build_pilot_input(raw, plan, t, outputs)
            assert payload["prior_generated_outputs"] == outputs
            assert payload["question"].startswith("QUESTION_")
            assert not any(x in json.dumps(payload) for x in ("SECRET", "untrusted", "UNSELECTED"))
            outputs[t["task_id"]] = "SYNTHETIC_TEST_ONLY"
        for t in left:
            assert build_pilot_input(raw, plan, t, {})["prior_generated_outputs"] == {}


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(sample(3)),
    lambda d: d["scene_0"].update(key_frames={}),
    lambda d: d["scene_0"]["key_frames"]["a_frame"]["QA"].pop("planning"),
    lambda d: d["scene_0"]["key_frames"]["a_frame"]["QA"].update(planning=[]),
    lambda d: d["scene_0"]["key_frames"]["a_frame"]["QA"].update(unknown=[]),
    lambda d: d["scene_0"]["key_frames"]["a_frame"]["image_paths"].pop("CAM_BACK"),
    lambda d: d["scene_0"]["key_frames"]["a_frame"]["QA"]["planning"][0].update(Q=" "),
])
def test_invalid_source_rejected_without_fallback_selection(mutation):
    data = sample()
    mutation(data)
    with pytest.raises(ValueError):
        prepare(data)


def test_answer_values_do_not_select_tasks():
    data = sample()
    _, before = prepare(data)
    data["scene_0"]["key_frames"]["a_frame"]["QA"]["planning"][0]["A"] = ""
    _, after = prepare(data)
    project = lambda p: [(t["scene_id"], t["frame_id"], t["question_ref"], t["arm"]) for t in p["tasks"]]
    assert project(before) == project(after)


@pytest.mark.parametrize("kind", ["missing", "extra", "future", "cross_arm", "cross_scene", "empty", "nonstring"])
def test_invalid_generated_context_rejected(kind):
    raw, plan = prepare()
    tasks = plan["tasks"]
    output = {tasks[4]["task_id"]: "generated perception"}
    if kind == "missing":
        output.clear()
    elif kind in ("extra", "future", "cross_arm", "cross_scene"):
        key = {"extra": "other", "future": tasks[6]["task_id"],
               "cross_arm": tasks[0]["task_id"], "cross_scene": tasks[12]["task_id"]}[kind]
        output[key] = "other output"
    else:
        output[tasks[4]["task_id"]] = " " if kind == "empty" else ["not one string"]
    with pytest.raises(ValueError):
        build_pilot_input(raw, plan, tasks[5], output)


def test_independent_arm_rejects_context():
    raw, plan = prepare()
    with pytest.raises(ValueError):
        build_pilot_input(raw, plan, plan["tasks"][1], {plan["tasks"][0]["task_id"]: "generated"})


@pytest.mark.parametrize("kind", ["budget", "bool_integer", "task", "extra_plan", "extra_task"])
def test_plan_and_task_tampering_rejected(kind):
    raw, plan = prepare()
    task = copy.deepcopy(plan["tasks"][0])
    if kind == "budget":
        plan["planned_model_calls"] = 99
    elif kind == "bool_integer":
        plan["actual_model_calls"] = False
    elif kind == "extra_plan":
        plan["extra"] = "bad"
    elif kind == "extra_task":
        task["A"] = "bad"
    else:
        task["question_ref"]["index"] = 1
    with pytest.raises(ValueError):
        build_pilot_input(raw, plan, task, {})


def test_source_drift_and_size_rejected():
    raw, plan = prepare()
    with pytest.raises(ValueError, match="identity"):
        build_pilot_input(raw + b" ", plan, plan["tasks"][0], {})
    with pytest.raises(ValueError, match="bounded"):
        build_pilot_plan(b" " * (MAX_BYTES + 1), "0" * 64)


@pytest.mark.parametrize("raw", [b'{"x": 1, "x": 2}', b'{"x": NaN}', b'[]', b'{}'])
def test_strict_json_validation(raw):
    with pytest.raises(ValueError):
        build_pilot_plan(raw, hashlib.sha256(raw).hexdigest())


def test_cli_stdout_metadata_only_and_invalid_hash():
    raw, expected = prepare()
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_drivelm_pilot.py")
    command = [sys.executable, "-B", script, "--input", "-", "--expected-sha256"]
    result = subprocess.run(command + [expected["source_sha256"]], input=raw, capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout) == expected
    assert b"SECRET" not in result.stdout and b"QUESTION_" not in result.stdout
    failed = subprocess.run(command + ["0" * 64], input=raw, capture_output=True)
    assert failed.returncode != 0 and failed.stdout == b""
