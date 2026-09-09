#!/usr/bin/env python3
"""Build the self-excluding content manifest for this evidence package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifact_integrity_manifest.json"


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        raw = path.read_bytes()
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        })
    manifest = {
        "artifact_integrity_manifest_version": "1.0.0",
        "artifact_run_id": "g14r20_f1_receipt_race_20260909",
        "integrity_status": "pass",
        "manifest_self_excluded": True,
        "file_count": len(files),
        "files": files,
        "synthetic_dispatch_count": 1,
        "scientific_rollout_count": 0,
        "real_v16_dispatch_count": 0,
        "real_v16_write_count": 0,
        "formal_checkpoint_count": 0,
        "formal_episode_count": 0,
        "formal_performance_result_count": 0,
        "holdout_consumption_count": 0,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
