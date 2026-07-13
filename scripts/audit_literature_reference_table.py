"""Audit the project literature table for duplicates and malformed references."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TABLE_PATH = ROOT_DIR / "docs" / "project" / "literature_reference_table.md"
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check literature-table titles, DOI/URL duplicates, links, and unverified rows."
    )
    parser.add_argument("--table_path", default=str(DEFAULT_TABLE_PATH))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fail_on_unverified", action="store_true")
    return parser.parse_args()


def normalize_title(title: str) -> str:
    return SPACE_RE.sub(" ", title.strip()).casefold()


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.casefold()
    if host in {"doi.org", "dx.doi.org"}:
        doi = unquote(parsed.path.lstrip("/")).strip().rstrip("/").casefold()
        return f"https://doi.org/{doi}"
    path = parsed.path.rstrip("/")
    normalized = f"{parsed.scheme.casefold()}://{host}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def extract_doi(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.netloc.casefold() not in {"doi.org", "dx.doi.org"}:
        return ""
    return unquote(parsed.path.lstrip("/")).strip().rstrip("/").casefold()


def parse_entries(table_path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    section = ""
    for line_number, raw_line in enumerate(
        table_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[1] == "论文":
            continue
        match = MARKDOWN_LINK_RE.search(cells[1])
        if match is None:
            continue
        title, url = match.groups()
        entries.append(
            {
                "line": str(line_number),
                "section": section,
                "direction": cells[0],
                "title": title.strip(),
                "url": url.strip(),
                "venue_year": cells[2],
                "row": line,
            }
        )
    return entries


def duplicate_groups(entries: list[dict[str, str]], key_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        value = entry[key_name]
        if value:
            grouped[value].append(entry)
    return [
        {
            "normalized_value": value,
            "count": len(group),
            "records": [
                {"line": int(item["line"]), "title": item["title"], "url": item["url"]}
                for item in group
            ],
        }
        for value, group in sorted(grouped.items())
        if len(group) > 1
    ]


def audit_table(table_path: Path) -> dict[str, Any]:
    entries = parse_entries(table_path)
    normalized_entries: list[dict[str, str]] = []
    invalid_links: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for entry in entries:
        parsed = urlparse(entry["url"])
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            invalid_links.append(
                {
                    "line": int(entry["line"]),
                    "title": entry["title"],
                    "url": entry["url"],
                    "reason": "expected absolute http(s) URL",
                }
            )
        if "待核验" in entry["row"]:
            unverified.append(
                {
                    "line": int(entry["line"]),
                    "title": entry["title"],
                    "venue_year": entry["venue_year"],
                }
            )
        normalized_entries.append(
            {
                **entry,
                "normalized_title": normalize_title(entry["title"]),
                "normalized_url": normalize_url(entry["url"]),
                "normalized_doi": extract_doi(entry["url"]),
            }
        )

    title_duplicates = duplicate_groups(normalized_entries, "normalized_title")
    doi_duplicates = duplicate_groups(normalized_entries, "normalized_doi")
    url_duplicates = duplicate_groups(normalized_entries, "normalized_url")
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "table_path": str(table_path.resolve()),
        "entry_count": len(entries),
        "passed": not title_duplicates and not doi_duplicates and not url_duplicates and not invalid_links,
        "title_duplicate_count": len(title_duplicates),
        "doi_duplicate_count": len(doi_duplicates),
        "url_duplicate_count": len(url_duplicates),
        "invalid_link_count": len(invalid_links),
        "unverified_count": len(unverified),
        "title_duplicates": title_duplicates,
        "doi_duplicates": doi_duplicates,
        "url_duplicates": url_duplicates,
        "invalid_links": invalid_links,
        "unverified": unverified,
        "link_validation_scope": "syntax_and_absolute_http_url_only; publisher HTTP reachability not tested",
    }
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "literature_table_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Literature Table Audit",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- entry_count: `{report['entry_count']}`",
        f"- passed: `{str(report['passed']).lower()}`",
        f"- title_duplicate_count: `{report['title_duplicate_count']}`",
        f"- doi_duplicate_count: `{report['doi_duplicate_count']}`",
        f"- url_duplicate_count: `{report['url_duplicate_count']}`",
        f"- invalid_link_count: `{report['invalid_link_count']}`",
        f"- unverified_count: `{report['unverified_count']}`",
        "",
        "`unverified` 是显式待核验清单，不会在默认模式下令审计失败。链接检查只验证绝对 HTTP(S) 结构，不声称已探测出版社页面可达性。",
    ]
    (output_dir / "literature_table_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    table_path = Path(args.table_path).resolve()
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    report = audit_table(table_path)
    write_report(report, Path(args.output_dir))
    print(json.dumps({key: report[key] for key in (
        "entry_count",
        "passed",
        "title_duplicate_count",
        "doi_duplicate_count",
        "url_duplicate_count",
        "invalid_link_count",
        "unverified_count",
    )}, ensure_ascii=False, indent=2))
    if not report["passed"] or (args.fail_on_unverified and report["unverified_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
