import copy
import hashlib
import json

import pytest

from scripts.build_driving_workflow_package import (
    build_package, consume_input, validate_interfaces, write_package, verify_package,
)
from scripts.prepare_drivelm_pilot import CAMERAS, STAGES


def fixture():
    data = {scene: {"key_frames": {"frame": {
        "QA": {stage: [{"Q": "QUESTION", "A": "SECRET_REFERENCE"}] for stage in STAGES},
        "key_object_infos": {"object": "SECRET_OBJECT"},
        "image_paths": {c: f"../nuscenes/samples/{c}/{scene}.jpg" for c in CAMERAS},
    }}} for scene in ("a", "b")}
    raw = json.dumps(data).encode()
    resources = {"annotation_sha256": hashlib.sha256(raw).hexdigest(), "annotation_bytes": len(raw),
                 "source_repository": "test", "source_revision": "fixed", "annotation_path": "sample.json",
                 "image_prefix": "samples/", "model_weight_metadata": {"revision": "fixed"},
                 "images": [{"path": f"{c}/{s}.jpg", "bytes": 1, "git_blob_sha1": "test"}
                            for c in CAMERAS for s in ("a", "b")]}
    return raw, resources


def test_single_workflow_reference_and_readback(tmp_path):
    raw, resources = fixture()
    package = build_package(raw, resources)
    assert package["scene_id"] == "a"
    assert len(package["tasks"]) == 8 and len(package["cameras"]) == 6
    assert sum(len(t["prior_task_ids"]) for t in package["tasks"]) == 6
    assert package["actual_model_calls"] == 0 and package["runtime_eligible"] is False
    assert not any(s in json.dumps(package) for s in ("SECRET", "QUESTION"))
    validation = validate_interfaces(raw, resources, package)
    assert validation["input_checks"] == 8 and validation["model_calls"] == 0
    root = tmp_path / "package"
    write_package(root, package, validation)
    verify_package(root)
    with pytest.raises(FileExistsError):
        write_package(root, package, validation)
    (root / "workflow.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_package(root)


@pytest.mark.parametrize("field", ["tasks", "cameras", "actual_model_calls", "runtime_eligible", "measured_costs"])
def test_package_mutation_rejected(field):
    raw, resources = fixture()
    package = build_package(raw, resources)
    task_id = package["tasks"][0]["task_id"]
    package[field] = "tampered"
    with pytest.raises(ValueError):
        consume_input(raw, resources, package, task_id, {})


def test_source_resources_and_context_rejected():
    raw, resources = fixture()
    package = build_package(raw, resources)
    with pytest.raises(ValueError):
        consume_input(raw + b" ", resources, package, package["tasks"][0]["task_id"], {})
    with pytest.raises(ValueError):
        consume_input(raw, resources, package, "other-scene", {})
    with pytest.raises(ValueError):
        consume_input(raw, resources, package, package["tasks"][5]["task_id"], {})
    broken = copy.deepcopy(resources)
    broken["images"].append(broken["images"][0])
    with pytest.raises(ValueError):
        build_package(raw, broken)
    broken = copy.deepcopy(resources)
    broken["images"].pop(0)
    with pytest.raises(ValueError):
        build_package(raw, broken)


def test_exact_membership(tmp_path):
    raw, resources = fixture()
    package = build_package(raw, resources)
    root = tmp_path / "package"
    write_package(root, package, validate_interfaces(raw, resources, package))
    (root / "extra").touch()
    with pytest.raises(ValueError):
        verify_package(root)
