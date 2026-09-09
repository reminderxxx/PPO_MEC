#!/usr/bin/env python3
"""Validate review formats and produce a self-excluding integrity manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


json_count = 0
xml_count = 0
for path in sorted(ROOT.iterdir()):
    if path.name in {"review_integrity_manifest.json", "review_validation.json"}:
        continue
    if path.suffix == ".json":
        json.loads(path.read_text())
        json_count += 1
    elif path.suffix == ".xml":
        ET.parse(path)
        xml_count += 1
for path in (ROOT / "review_checks.py", ROOT / "validate_review.py"):
    compile(path.read_text(), str(path), "exec")

validation = {
    "status": "pass",
    "json_count": json_count,
    "xml_count": xml_count,
    "production_private_key_generated": False,
    "production_trust_installed": False,
    "continuation_approval_issued": False,
    "real_execution_authorized": False,
}
(ROOT / "review_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

files = []
for path in sorted(ROOT.iterdir()):
    if path.is_file() and path.name != "review_integrity_manifest.json":
        files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path)})
manifest = {"version": "1.0.0", "files": files}
(ROOT / "review_integrity_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
for row in files:
    path = ROOT / row["path"]
    assert path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"]
print(json.dumps({**validation, "manifest_files": len(files)}, sort_keys=True))
