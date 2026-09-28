"""Read-only, bounded DriveLM annotation qualification; never infer graph edges."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


RELATION_FIELDS = ("C", "con_up", "con_down", "cluster", "layer")
STAGES = ("perception", "prediction", "planning", "behavior")


def build_preview(raw: bytes) -> dict:
    """Derive a proposed task template, not a recovered DriveLM dependency graph."""
    audit = audit_sample(raw)
    data = json.loads(raw)
    workflows = []
    for scene_id in sorted(data):
        for frame_id, frame in sorted(data[scene_id]["key_frames"].items()):
            unknown = set(frame["QA"]) - set(STAGES)
            if unknown:
                raise ValueError(f"unmapped QA categories: {sorted(unknown)}")
            present = [s for s in STAGES if frame["QA"].get(s)]
            nodes = [{
                "node_id": stage,
                "question_refs": [{"category": stage, "index": i}
                                  for i in range(len(frame["QA"][stage]))],
                "predecessors": present[:idx],
                "model_binding": None, "adapter_binding": None,
                "measured_latency_ms": None, "measured_state_bytes": None,
            } for idx, stage in enumerate(present)]
            workflows.append({
                "scene_id": scene_id, "frame_id": frame_id,
                "complete_four_stage_template": len(present) == len(STAGES),
                "nodes": nodes,
                "edges": [[p, n["node_id"]] for n in nodes for p in n["predecessors"]],
            })
    return {
        "schema_version": "drivelm_derived_preview_0.1",
        "source_sha256": audit["source_sha256"],
        "source_bytes": audit["source_bytes"],
        "edge_provenance": "researcher_defined_all_prior_stage_context_not_observed",
        "selection_rule": "all_frames_sorted_scene_frame_no_outcome_filter",
        "workflows": workflows,
        "runtime_eligible": False, "performance_evidence": False,
        "unresolved": ["licensed_image_inputs", "model_and_adapter_bindings",
                       "task_quality", "measured_costs", "mobility_mapping"],
    }


def build_stage_input(raw: bytes, source_sha256: str, scene_id: str,
                      frame_id: str, stage: str, generated: dict) -> dict:
    """Construct a testable prompt payload; never copy reference A/object labels.

    generated is caller-supplied inference output. This constructor checks shape,
    not its scientific provenance. It does not load images or invoke a model.
    """
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError("source byte identity mismatch")
    preview = build_preview(raw)
    workflow = next((w for w in preview["workflows"]
                     if w["scene_id"] == scene_id and w["frame_id"] == frame_id), None)
    if workflow is None:
        raise ValueError("unknown scene/frame")
    node = next((n for n in workflow["nodes"] if n["node_id"] == stage), None)
    if node is None or not isinstance(generated, dict) or set(generated) != set(node["predecessors"]):
        raise ValueError("missing, extra or future stage output")
    frame = json.loads(raw)[scene_id]["key_frames"][frame_id]
    for predecessor, outputs in generated.items():
        if (not isinstance(outputs, list) or len(outputs) != len(frame["QA"][predecessor])
                or not all(isinstance(x, str) and x.strip() for x in outputs)):
            raise ValueError("generated output count/type mismatch")
    return {
        "source_sha256": source_sha256, "scene_id": scene_id, "frame_id": frame_id,
        "stage": stage,
        "questions": [r["Q"] for r in frame["QA"][stage]],
        "prior_generated_outputs": {p: list(generated[p]) for p in node["predecessors"]},
        "image_binding": {"status": "unresolved", "scene_id": scene_id, "frame_id": frame_id},
        "reference_answers_included": False, "runtime_eligible": False,
    }


def audit_sample(raw: bytes) -> dict:
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    data = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or not data:
        raise ValueError("expected nonempty scene mapping")
    categories, present, null, nonnull = (Counter() for _ in range(4))
    frame_count = qa_count = 0
    for scene in data.values():
        if not isinstance(scene, dict) or not isinstance(scene.get("key_frames"), dict):
            raise ValueError("scene lacks key_frames mapping")
        for frame in scene["key_frames"].values():
            if not isinstance(frame, dict) or not isinstance(frame.get("QA"), dict):
                raise ValueError("frame lacks QA mapping")
            frame_count += 1
            for category, rows in frame["QA"].items():
                if not isinstance(rows, list):
                    raise ValueError("QA category must contain a list")
                categories[category] += len(rows)
                for row in rows:
                    if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("Q", "A")):
                        raise ValueError("QA record lacks string Q/A")
                    qa_count += 1
                    for field in RELATION_FIELDS:
                        if field in row:
                            present[field] += 1
                            if row[field] is None:
                                null[field] += 1
                            else:
                                nonnull[field] += 1
    if not qa_count:
        raise ValueError("sample has no QA records")
    return {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw), "scene_count": len(data),
        "frame_count": frame_count, "qa_count": qa_count,
        "categories": dict(sorted(categories.items())),
        "relation_fields": {
            field: {"present": present[field], "missing": qa_count - present[field],
                    "null": null[field], "nonnull": nonnull[field]}
            for field in RELATION_FIELDS
        },
        "observed_dependency_graph_verified": False,
        "graph_status": "RELATION_FIELDS_REQUIRE_SEMANTIC_AUDIT" if any(nonnull.values())
                        else "NO_NON_NULL_RELATION_FIELDS",
        "inferred_edge_count": 0,
        "scope": "annotation_structure_only_not_model_or_dataset_validation",
        "performance_evidence": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local JSON or '-' for stdin")
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--preview", action="store_true", help="emit proposed workflow references, not observed edges")
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")
    if args.input == "-":
        raw = sys.stdin.buffer.read(args.max_bytes + 1)
    else:
        with Path(args.input).open("rb") as stream:
            raw = stream.read(args.max_bytes + 1)
    if len(raw) > args.max_bytes:
        parser.error("input exceeds bounded sample size")
    report = build_preview(raw) if args.preview else audit_sample(raw)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
