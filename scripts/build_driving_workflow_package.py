"""Build a single-scene reference package, not a model execution or data release.

Raw licensed Q/A and images are not copied. The consumer rebuilds the package
from exact source bytes before constructing an input with generated context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

try:
    from .prepare_drivelm_pilot import MAX_BYTES, build_pilot_plan, build_pilot_input
except ImportError:
    from prepare_drivelm_pilot import MAX_BYTES, build_pilot_plan, build_pilot_input

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_PATH = ROOT / "configs/experiment/drivelm_pilot_resources.json"


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       indent=2) + "\n").encode()


def build_package(raw, resources):
    plan = build_pilot_plan(raw, resources["annotation_sha256"])
    if len(raw) != resources["annotation_bytes"]:
        raise ValueError("source size mismatch")
    scene = min(t["scene_id"] for t in plan["tasks"])
    tasks = [t for t in plan["tasks"] if t["scene_id"] == scene]
    frame_id = tasks[0]["frame_id"]
    frame = json.loads(raw)[scene]["key_frames"][frame_id]
    inventory = {r["path"]: r for r in resources["images"]}
    if len(inventory) != len(resources["images"]):
        raise ValueError("duplicate image resource")
    cameras = []
    for camera in tasks[0]["image_camera_names"]:
        path = frame["image_paths"][camera]
        prefix = "../nuscenes/samples/"
        if not path.startswith(prefix):
            raise ValueError("unknown source image prefix")
        relative = path[len(prefix):]
        if relative not in inventory or relative.split("/")[0] != camera:
            raise ValueError("image not in camera-specific allowlist")
        if ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError("unsafe resource reference")
        row = inventory[relative]
        cameras.append({"camera": camera, **row,
                        "content_sha256": None, "decoded_size": None,
                        "status": "reference_only_not_downloaded"})
    return {
        "schema_version": "driving_workflow_reference_package_0.1",
        "status": "REFERENCE_PACKAGE_NOT_INFERENCE_VALIDATED",
        "source": {k: resources[k] for k in ("source_repository", "source_revision",
                                             "annotation_path", "annotation_sha256",
                                             "annotation_bytes", "image_prefix")},
        "selection": "first_lexicographic_scene_then_frame_Q0_no_outcome_filter",
        "scene_id": scene, "frame_id": frame_id,
        "tasks": tasks, "cameras": cameras,
        "edge_provenance": plan["edge_provenance"],
        "model_candidate": resources["model_weight_metadata"],
        "planned_model_calls": 8, "actual_model_calls": 0,
        "decoding_plan": {"do_sample": False, "max_new_tokens": 64, "batch_size": 1},
        "adapter_binding": None, "mobility_binding": None, "measured_costs": None,
        "performance_evidence": False, "runtime_eligible": False,
        "license_status": "image_terms_confirmation_pending_no_raw_redistribution",
        "expansion_requires": ["real_model_execution", "blinded_task_quality_review",
                               "output_dependency_check", "state_restore_equivalence",
                               "measured_costs", "license_clearance"],
    }


def consume_input(raw, resources, package, task_id, generated):
    if encode(package) != encode(build_package(raw, resources)):
        raise ValueError("package differs from source-derived package")
    task = next((t for t in package["tasks"] if t["task_id"] == task_id), None)
    if task is None:
        raise ValueError("task outside single-scene package")
    plan = build_pilot_plan(raw, resources["annotation_sha256"])
    return build_pilot_input(raw, plan, task, generated)


def validate_interfaces(raw, resources, package):
    """Explicit synthetic context traversal; never call a model or claim quality."""
    outputs = {}
    for task in package["tasks"]:
        generated = {key: outputs[key] for key in task["prior_task_ids"]}
        payload = consume_input(raw, resources, package, task["task_id"], generated)
        assert payload["prior_generated_outputs"] == generated
        outputs[task["task_id"]] = "SYNTHETIC_INTERFACE_TEST_NOT_MODEL_OUTPUT"
    return {"status": "interface_pass", "input_checks": len(outputs),
            "model_calls": 0, "quality_assessed": False, "performance_evidence": False}


def write_package(root, package, validation):
    card = ("# Driving AI workflow reference mini-package\n\n"
            "One scene, six camera references, four stages, two matched arms (8 planned calls).\n"
            "References only: no source images, questions, answers or model outputs are included.\n"
            "Edges are researcher-defined, not observed executable dependencies.\n"
            "This is not an autonomous-driving controller, independent holdout or released dataset.\n"
            "Actual model calls: 0. Synthetic input traversal is not inference/quality validation.\n"
            "Expansion requires real inference, blinded quality, dependency/restore tests and costs.\n"
            "No adapter, mobility, network or migration cost is measured in this package.\n")
    files = {"workflow.json": encode(package), "interface_validation.json": encode(validation),
             "data_card.md": card.encode()}
    root.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        with (root / name).open("xb") as stream:
            stream.write(payload)
    manifest = {"files": [{"path": name, "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()}
                           for name, payload in sorted(files.items())],
                "self_excluded": "integrity_manifest.json"}
    with (root / "integrity_manifest.json").open("xb") as stream:
        stream.write(encode(manifest))
    verify_package(root)


def verify_package(root):
    manifest = json.loads((root / "integrity_manifest.json").read_bytes())
    expected = {"workflow.json", "interface_validation.json", "data_card.md"}
    if ({r["path"] for r in manifest["files"]} != expected or len(manifest["files"]) != 3
            or {p.name for p in root.iterdir()} != expected | {"integrity_manifest.json"}):
        raise ValueError("inventory membership mismatch")
    for row in manifest["files"]:
        path = root / row["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("not a regular payload")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("package payload mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--fetch-official-sample", action="store_true",
                        help="bounded in-memory official annotation fetch; never saves raw QA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = json.loads(RESOURCE_PATH.read_bytes())
    if args.fetch_official_sample:
        url = (f"https://raw.githubusercontent.com/{resources['source_repository']}/"
               f"{resources['source_revision']}/{resources['annotation_path']}")
        with urllib.request.urlopen(url, timeout=30) as stream:
            raw = stream.read(MAX_BYTES + 1)
    else:
        with args.input.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    package = build_package(raw, resources)
    validation = validate_interfaces(raw, resources, package)
    write_package(args.output, package, validation)
    print(json.dumps({"output": str(args.output), "package_status": package["status"], **validation}))


if __name__ == "__main__":
    main()
