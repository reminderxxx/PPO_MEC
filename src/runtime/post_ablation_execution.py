"""Read-only G14E06 handoff validation for the remaining evaluation phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.evaluators.formal_cell_transaction import (
    artifact_inventory,
    stable_cell_id,
    validate_producer_integrity_manifests,
    verify_persisted_committed_phase,
)
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ORIGINAL_RUN_ROOT,
    audit_original_recovery_source,
    validate_recovery_handoff_manifest,
)

HANDOFF_PATH = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/"
    "typed_model_cache_restricted_recovery/"
    "typed_model_cache_restricted_recovery_20260920_g14r20_i5e_pending/"
    "ablation_recovery_handoff.json"
)
POST_PHASES = (
    "formal_support", "formal_scalability", "formal_statistics",
    "formal_gate", "complete_without_holdout",
)


class PostAblationError(ValueError):
    pass


def audit_handoff(path: str | Path = HANDOFF_PATH, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Reopen both source ledgers and verify exact external cell payloads."""

    target = Path(path)
    if target != HANDOFF_PATH or target.is_symlink() or not target.is_file():
        raise PostAblationError("unreviewed or missing G14E06 handoff")
    payload = json.loads(target.read_text(encoding="utf-8"))
    validate_recovery_handoff_manifest(payload)
    if expected_sha256 is not None and payload["handoff_sha256"] != expected_sha256:
        raise PostAblationError("handoff request hash drift")
    original = audit_original_recovery_source()["external_committed_cells"]
    source_rows = [row for row in payload["cells"] if row.get("origin") == "external_original_run"]
    if source_rows != original:
        raise PostAblationError("six original cell references drift")
    recovery_root = target.parent
    recovered = verify_persisted_committed_phase(recovery_root, phase="formal_ablation")
    recovered_rows = [row for row in payload["cells"] if row.get("origin") == "new_recovery_execution"]
    if len(recovered) != 2 or len(recovered_rows) != 2:
        raise PostAblationError("recovered ablation membership drift")
    for row, record in zip(recovered_rows, recovered):
        path = Path(row["committed_path"])
        if path.is_symlink() or recovery_root not in path.parents:
            raise PostAblationError("recovered cell escapes immutable root")
        inventory = artifact_inventory(path)
        producer = validate_producer_integrity_manifests(path)
        if any((
            row["cell_id"] != record["cell_id"],
            row["cell_id"] != stable_cell_id(row["phase"], row["coordinates"]),
            row["coordinates"] != record["coordinates"],
            row["committed_path"] != record["committed_path"],
            row["recovery_attempt"] != record["attempt"],
            row["transaction_inventory_sha256"] != canonical_sha256(inventory),
            row["transaction_file_count"] != len(inventory),
            row["committed_marker_sha256"] != file_sha256(path / "committed_marker.json"),
            row["producer_integrity"] != producer,
        )):
            raise PostAblationError("recovered cell identity or dual integrity drift")
    if sorted(row["recovery_attempt"] for row in recovered_rows) != [1, 3]:
        raise PostAblationError("ablation attempt lineage drift")
    expected = {"formal_cache_policy": 3, "formal_controller": 3, "formal_ablation": 2}
    for phase, count in expected.items():
        if sum(row["phase"] == phase for row in payload["cells"]) != count:
            raise PostAblationError("external phase cell count drift")
    controller_paths = sorted(
        str(next(Path(row["committed_path"]).rglob("benchmark_rows.csv")).resolve())
        for row in payload["cells"] if row["phase"] == "formal_controller"
    )
    compatibility = payload["statistics_consumer_compatibility"]
    if (compatibility.get("formal_controller_row_paths") != controller_paths
            or compatibility.get("formal_controller_row_count") != 3):
        raise PostAblationError("controller statistics input binding drift")
    if any("/ABSOLUTE/" in str(path) or "{" in str(path) for path in controller_paths):
        raise PostAblationError("unresolved executable controller row path")
    return {"status": "pass", "handoff_sha256": payload["handoff_sha256"],
            "cells": payload["cells"], "controller_rows": controller_paths,
            "source_roots": [str(ORIGINAL_RUN_ROOT), str(recovery_root)]}


def require_bound_handoff(input_root: str | Path, supplied: str | Path) -> dict[str, Any]:
    root = Path(input_root)
    reference_path = root / "post_ablation_handoff_reference.json"
    if reference_path.is_symlink() or not reference_path.is_file():
        raise PostAblationError("post-ablation handoff reference is missing")
    reference: Mapping[str, Any] = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("path") != str(supplied):
        raise PostAblationError("consumer handoff path differs from run reference")
    return audit_handoff(supplied, expected_sha256=reference.get("sha256"))
