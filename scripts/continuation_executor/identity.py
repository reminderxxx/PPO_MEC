"""Strict, non-writing executor and artifact identity primitives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


class ContinuationError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(text):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ContinuationError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          ContinuationError("non-finite JSON: " + x)))


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def absolute_path(value):
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts or str(p) != str(value):
        raise ContinuationError("non-canonical path: " + str(value))
    if any(part.is_symlink() for part in (p, *p.parents)):
        raise ContinuationError("symlink path: " + str(value))
    return p


def within(value, root):
    p, root = absolute_path(value), absolute_path(root)
    if p != root and root not in p.parents:
        raise ContinuationError("path escapes approved root: " + str(value))
    return p


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


def verify_file(row):
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise ContinuationError("file identity fields")
    path = absolute_path(row["path"])
    if not path.is_file() or path.stat().st_size != row["size_bytes"] or file_hash(path) != row["sha256"]:
        raise ContinuationError("file identity drift: " + str(path))
    return path


def verify_executor(identity, root):
    """Identity lives outside source, avoiding a commit/self-hash cycle."""
    root = absolute_path(root)
    if set(identity) != {"version", "commit", "git_tree", "files"} or identity["version"] != "1.0.0":
        raise ContinuationError("executor identity schema")
    if not re.fullmatch("[0-9a-f]{40}", identity["commit"]):
        raise ContinuationError("executor commit must be exact")
    if git(root, "rev-parse", "HEAD") != identity["commit"] or git(root, "rev-parse", "HEAD^{tree}") != identity["git_tree"]:
        raise ContinuationError("executor fixed commit/tree drift")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ContinuationError("executor source must be clean")
    expected = {"scripts/execute_fixed_commit_continuation.py", "scripts/run_fixed_commit_continuation_acceptance.py", "src/runtime/fixed_commit_continuation.py"}
    expected.update(p.relative_to(root).as_posix() for p in (root / "scripts/continuation_executor").glob("*.py"))
    rows = identity["files"]
    if not isinstance(rows, list) or len(rows) != len(expected) or {r.get("path") for r in rows} != expected:
        raise ContinuationError("executor file manifest must be complete and unique")
    for row in rows:
        if set(row) != {"path", "sha256"} or file_hash(within(str(root / row["path"]), root)) != row["sha256"]:
            raise ContinuationError("executor file drift")
    return {"executor_identity_sha256": digest(identity), "fixed_commit": identity["commit"],
            "file_count": len(rows), "current_branch_observation": git(root, "rev-parse", "--abbrev-ref", "HEAD")}


def validate_a_proposal(proposal):
    """Load A by filename so importing it cannot shadow old scientific src."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "src/runtime/fixed_commit_continuation.py"
    spec = importlib.util.spec_from_file_location("continuation_a_schema_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_structure(proposal)
