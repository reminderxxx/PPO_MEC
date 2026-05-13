from __future__ import annotations

from pathlib import Path

from src.data.workflow import alibaba_dag_parser
from src.data.workflow.alibaba_dag_parser import AlibabaDAGParser


TEST_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "tmp_validation" / "alibaba_dag_parser_tests"


def test_safe_numeric_parsing_uses_builtin_types(monkeypatch) -> None:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TEST_ROOT / "batch_task.csv"
    csv_path.write_text("", encoding="utf-8")
    parser = AlibabaDAGParser(csv_path)
    monkeypatch.setattr(alibaba_dag_parser, "float", "shadowed", raising=False)
    monkeypatch.setattr(alibaba_dag_parser, "int", "shadowed", raising=False)
    monkeypatch.setattr(alibaba_dag_parser, "str", "shadowed", raising=False)

    assert parser._safe_int("1,024.0") == 1024
    assert parser._safe_float("2,048.5") == 2048.5


def test_collect_jobs_stops_after_job_limit_without_parsing_later_dirty_rows() -> None:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TEST_ROOT / "batch_task_dirty_tail.csv"
    csv_path.write_text(
        "\n".join(
            [
                "A1,1,j_1,1,Terminated,1,2,100,0.1",
                "A2,1,j_1,1,Terminated,2,3,100,0.2",
                "B1,1,j_2,1,Terminated,1,2,100,0.1",
                "B2,1,j_2,1,Terminated,2,3,100,0.2",
                "C1,1,j_3,1,Terminated,1,2,100,not_a_number",
            ]
        ),
        encoding="utf-8",
    )
    parser = AlibabaDAGParser(csv_path)

    jobs = parser.collect_jobs(limit_jobs=2, min_tasks=1, max_tasks=10)

    assert sorted(jobs) == ["j_1", "j_2"]
