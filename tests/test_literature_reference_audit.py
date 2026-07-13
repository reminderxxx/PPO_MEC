from __future__ import annotations

from pathlib import Path

from scripts.audit_literature_reference_table import audit_table


HEADER = "| 方向 | 论文 | Venue / Year | 可提供的参考点 | PPO_MEC 的优化点 / 差异点 | 论文写作位置 |\n|---|---|---:|---|---|---|\n"


def _write_table(path: Path, rows: list[str]) -> Path:
    path.write_text("# Table\n\n## Section\n\n" + HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_literature_audit_accepts_unique_rows_and_reports_unverified(tmp_path: Path) -> None:
    table = _write_table(
        tmp_path / "literature.md",
        [
            "| DAG | [Paper A](https://doi.org/10.1000/a) | TMC, 2025 | A | B | Related Work |",
            "| cache | [Paper B](https://arxiv.org/abs/1234.5678) | venue 待核验 | A | B | Discussion |",
        ],
    )

    report = audit_table(table)

    assert report["passed"] is True
    assert report["entry_count"] == 2
    assert report["unverified_count"] == 1


def test_literature_audit_detects_title_doi_url_duplicates_and_invalid_link(tmp_path: Path) -> None:
    table = _write_table(
        tmp_path / "literature.md",
        [
            "| DAG | [Paper A](https://doi.org/10.1000/a) | TMC, 2025 | A | B | Related Work |",
            "| cache | [ paper   a ](https://doi.org/10.1000/A/) | TMC, 2025 | A | B | Related Work |",
            "| cache | [Paper C](ftp://example.org/paper) | TMC, 2025 | A | B | Related Work |",
        ],
    )

    report = audit_table(table)

    assert report["passed"] is False
    assert report["title_duplicate_count"] == 1
    assert report["doi_duplicate_count"] == 1
    assert report["url_duplicate_count"] == 1
    assert report["invalid_link_count"] == 1
