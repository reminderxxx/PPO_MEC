import hashlib
import json

import pytest

from scripts.audit_drivelm_workflow_sample import STAGES, build_preview, build_stage_input


def fixture():
    return json.dumps({"scene": {"key_frames": {"frame": {
        "QA": {s: [{"Q": f"Question {s}", "A": "SECRET_REFERENCE"}] for s in STAGES},
        "key_object_infos": {"hidden": "SECRET_OBJECT_LABEL"},
    }}}}).encode()


def test_preview_complete_and_not_observed():
    p = build_preview(fixture())
    assert len(p["workflows"]) == 1
    w = p["workflows"][0]
    assert len(w["nodes"]) == 4 and len(w["edges"]) == 6
    assert w["complete_four_stage_template"]
    assert not p["runtime_eligible"] and not p["performance_evidence"]
    assert "not_observed" in p["edge_provenance"]
    assert all(n["model_binding"] is None for n in w["nodes"])
    assert "SECRET" not in json.dumps(p)


def test_generated_outputs_reach_downstream_without_reference_labels():
    raw = fixture()
    sha = hashlib.sha256(raw).hexdigest()
    outputs = {}
    for stage in STAGES:
        payload = build_stage_input(raw, sha, "scene", "frame", stage, outputs)
        assert payload["prior_generated_outputs"] == outputs
        assert "SECRET" not in json.dumps(payload)
        outputs[stage] = [f"SYNTHETIC_TEST_OUTPUT_{stage}"]
    changed = {"perception": ["altered upstream result"]}
    p = build_stage_input(raw, sha, "scene", "frame", "prediction", changed)
    assert p["prior_generated_outputs"] == changed


@pytest.mark.parametrize("outputs", [{}, {"perception": []}, {"perception": [None]},
                                      {"perception": [""]}, {"planning": ["future"]},
                                      {"perception": ["ok"], "planning": ["future"]}])
def test_invalid_dependencies_rejected(outputs):
    raw = fixture()
    with pytest.raises(ValueError):
        build_stage_input(raw, hashlib.sha256(raw).hexdigest(), "scene", "frame", "prediction", outputs)


def test_source_drift_rejected():
    with pytest.raises(ValueError, match="identity"):
        build_stage_input(fixture(), "0" * 64, "scene", "frame", "perception", {})


def test_missing_stage_preserved_not_fabricated():
    d = json.loads(fixture())
    d["scene"]["key_frames"]["frame"]["QA"]["planning"] = []
    w = build_preview(json.dumps(d).encode())["workflows"][0]
    assert not w["complete_four_stage_template"]
    assert "planning" not in [n["node_id"] for n in w["nodes"]]


def test_unknown_category_rejected_not_dropped():
    d = json.loads(fixture())
    d["scene"]["key_frames"]["frame"]["QA"]["unknown"] = []
    with pytest.raises(ValueError, match="unmapped"):
        build_preview(json.dumps(d).encode())
