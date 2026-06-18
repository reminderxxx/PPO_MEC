"""Audit experiment artifact references and generate a SHA-256 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
PATH_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".pt",
    ".pth",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate artifact references and generate a SHA-256 inventory."
    )
    parser.add_argument("--run_root", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_root", default=str(ROOT_DIR))
    return parser.parse_args()


def iter_strings(value: Any, *, required: bool = True) -> Iterable[tuple[str, bool]]:
    if isinstance(value, str):
        yield value, required
    elif isinstance(value, dict):
        child_required = required and value.get("exists") is not False
        for item in value.values():
            yield from iter_strings(item, required=child_required)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item, required=required)


def looks_like_path(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith("--") or "\n" in text:
        return False
    # Provenance descriptions can embed several labelled paths in one string;
    # the complete description is not itself a filesystem reference.
    if ";" in text or "=/" in text:
        return False
    path = Path(text).expanduser()
    return (
        path.is_absolute()
        or path.suffix.lower() in PATH_SUFFIXES
        or text.startswith(("artifacts/", "configs/", "data/", "docs/", "scripts/", "src/"))
    )


def resolve_reference(value: str, *, source_path: Path, repo_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (source_path.parent / candidate).resolve()


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path.resolve()
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and not child.is_symlink():
                yield child.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def audit(run_roots: list[Path], *, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    normalized_roots = [path.resolve() for path in run_roots]
    missing_roots = [str(path) for path in normalized_roots if not path.exists()]
    if missing_roots:
        raise FileNotFoundError("missing run roots: " + ", ".join(missing_roots))

    root_files = {file for root in normalized_roots for file in iter_files(root)}
    json_errors: list[dict[str, str]] = []
    references: dict[Path, set[Path]] = {}
    missing_references: list[dict[str, str]] = []
    for source_path in sorted(root_files):
        if source_path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            json_errors.append({"source": display_path(source_path, repo_root), "error": str(exc)})
            continue
        for value, required in iter_strings(payload):
            if not looks_like_path(value):
                continue
            resolved = resolve_reference(value, source_path=source_path, repo_root=repo_root)
            if resolved.exists():
                references.setdefault(resolved, set()).add(source_path)
            elif required:
                missing_references.append(
                    {
                        "source": display_path(source_path, repo_root),
                        "reference": value,
                        "resolved": display_path(resolved, repo_root),
                    }
                )

    dependency_paths = {
        path
        for path in references
        if not any(path == root.parent or path in root.parents for root in normalized_roots)
    }
    referenced_files = {file for path in dependency_paths for file in iter_files(path)}
    all_files = root_files | referenced_files
    output_dir = output_dir.resolve()
    all_files = {path for path in all_files if output_dir not in path.parents}

    inventory_rows = [
        {"sha256": sha256_file(path), "path": display_path(path, repo_root)}
        for path in sorted(all_files, key=lambda item: display_path(item, repo_root))
    ]
    external_references = [
        {
            "path": display_path(path, repo_root),
            "referenced_by_count": len(sources),
            "referenced_by": sorted(display_path(source, repo_root) for source in sources)[:20],
        }
        for path, sources in sorted(references.items(), key=lambda item: display_path(item[0], repo_root))
        if path not in root_files
        and not any(root in path.parents or path in root.parents for root in normalized_roots)
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "run_roots": [display_path(path, repo_root) for path in normalized_roots],
        "passed": not missing_references and not json_errors,
        "root_file_count": len(root_files),
        "referenced_file_count": len(referenced_files),
        "inventory_file_count": len(inventory_rows),
        "external_reference_count": len(external_references),
        "missing_reference_count": len(missing_references),
        "json_error_count": len(json_errors),
        "external_references": external_references,
        "missing_references": missing_references,
        "json_errors": json_errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact_integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_lines = [f"{row['sha256']}  {row['path']}" for row in inventory_rows]
    (output_dir / "sha256_manifest.txt").write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = audit(
        [Path(path) for path in args.run_root],
        output_dir=Path(args.output_dir),
        repo_root=Path(args.repo_root).resolve(),
    )
    summary = {
        key: report[key]
        for key in (
            "created_at",
            "run_roots",
            "passed",
            "root_file_count",
            "referenced_file_count",
            "inventory_file_count",
            "external_reference_count",
            "missing_reference_count",
            "json_error_count",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
