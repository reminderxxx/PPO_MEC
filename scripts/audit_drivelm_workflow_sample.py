"""Read-only, bounded DriveLM annotation qualification; never infer graph edges."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


RELATION_FIELDS = ("C", "con_up", "con_down", "cluster", "layer")


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
    print(json.dumps(audit_sample(raw), ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
