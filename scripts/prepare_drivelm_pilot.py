"""Prepare a bounded, metadata-only DriveLM pilot; never execute inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    from .audit_drivelm_workflow_sample import STAGES, audit_sample
except ImportError:  # Direct script entry point.
    from audit_drivelm_workflow_sample import STAGES, audit_sample


MAX_BYTES = 2_000_000
ARMS = ("independent", "generated_context_same_frame")
CAMERAS = ("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT",
           "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def build_pilot_plan(raw: bytes, expected_sha256: str) -> dict:
    """Bind deterministic Q0 references to exact source bytes, without Q/A text."""
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("input exceeds bounded sample size or is not bytes")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source byte identity mismatch")
    audit = audit_sample(raw)
    data = json.loads(raw)
    if not 1 <= len(data) <= 2:
        raise ValueError("pilot requires one or two scenes; no scene subsampling")
    tasks = []
    for scene_id in sorted(data):
        frames = data[scene_id]["key_frames"]
        if not frames:
            raise ValueError("scene has no frames")
        frame_id = sorted(frames)[0]
        frame = frames[frame_id]
        if set(frame["QA"]) != set(STAGES):
            raise ValueError("selected frame requires exactly four known stages")
        if any(not frame["QA"][s] or not frame["QA"][s][0]["Q"].strip() for s in STAGES):
            raise ValueError("selected frame missing nonempty stage Q0")
        images = frame.get("image_paths")
        if (not isinstance(images, dict) or set(images) != set(CAMERAS)
                or any(not isinstance(p, str) or not p.strip() for p in images.values())):
            raise ValueError("selected frame requires all six camera identities")
        for arm in ARMS:
            previous = []
            for stage in STAGES:
                identity = [expected_sha256, scene_id, frame_id, arm, stage, 0]
                task_id = hashlib.sha256(_canonical(identity).encode()).hexdigest()
                tasks.append({
                    "task_id": task_id, "scene_id": scene_id, "frame_id": frame_id,
                    "arm": arm, "question_ref": {"category": stage, "index": 0},
                    "prior_task_ids": list(previous) if arm == ARMS[1] else [],
                    "image_camera_names": list(CAMERAS),
                })
                previous.append(task_id)
    return {
        "schema_version": "drivelm_pilot_plan_0.1",
        "source_sha256": audit["source_sha256"], "source_bytes": len(raw),
        "scene_count": len(data), "tasks": tasks,
        "selection_rule": "all_scenes_first_lexicographic_frame_each_stage_index_0_no_answer_filter",
        "edge_provenance": "researcher_defined_all_prior_stage_context_not_observed",
        "question_reference_provenance": "derived_references_to_source_Q0",
        "arms": list(ARMS), "stage_order": list(STAGES),
        "planned_model_calls": len(tasks), "model_call_cap": 16,
        "actual_model_calls": 0, "performance_evidence": False,
        "runtime_eligible": False, "model_binding": None, "adapter_binding": None,
        "image_binding": None, "measured_costs": None,
        "scope": "chain_interface_pilot_not_independent_statistical_evidence",
        "unresolved": ["licensed_six_camera_inputs_preserving_coordinates", "model_version_and_license",
                       "isolated_runtime", "token_time_image_costs", "task_quality",
                       "generated_output_provenance", "question_embedded_annotation_hints"],
    }


def build_pilot_input(raw: bytes, plan: dict, task: dict, generated: dict) -> dict:
    """Pure payload builder; generated maps prior task IDs to nonempty Q0 outputs.

    This checks declared identity and shape, not actual model execution provenance.
    Raw image paths are never returned, resolved, read or joined. No reference A or
    key_object_infos are copied. Source Q may itself contain annotation hints.
    """
    if not isinstance(plan, dict) or not isinstance(task, dict):
        raise ValueError("plan/task must be mappings")
    rebuilt = build_pilot_plan(raw, plan.get("source_sha256"))
    if _canonical(plan) != _canonical(rebuilt):
        raise ValueError("plan differs from exact source-derived plan")
    matches = [t for t in rebuilt["tasks"] if t["task_id"] == task.get("task_id")]
    if len(matches) != 1 or _canonical(task) != _canonical(matches[0]):
        raise ValueError("task differs from exact source-derived task")
    prior = task["prior_task_ids"]
    if not isinstance(generated, dict) or set(generated) != set(prior):
        raise ValueError("missing, extra, future or cross-arm/frame generated output")
    if any(not isinstance(v, str) or not v.strip() for v in generated.values()):
        raise ValueError("generated Q0 output must be a nonempty string")
    frame = json.loads(raw)[task["scene_id"]]["key_frames"][task["frame_id"]]
    return {
        "source_sha256": plan["source_sha256"], "task_id": task["task_id"],
        "scene_id": task["scene_id"], "frame_id": task["frame_id"], "arm": task["arm"],
        "question": frame["QA"][task["question_ref"]["category"]][0]["Q"],
        "prior_generated_outputs": {key: generated[key] for key in prior},
        "image_binding": {"status": "unresolved", "camera_names": list(CAMERAS)},
        "reference_answers_included": False, "runtime_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local JSON or '-' for stdin")
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.input == "-":
            raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        else:
            with Path(args.input).open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
        plan = build_pilot_plan(raw, args.expected_sha256)
    except (OSError, ValueError) as error:
        # Do not echo source-derived text (e.g. duplicate keys) in diagnostics.
        parser.error(f"pilot input rejected ({type(error).__name__})")
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
