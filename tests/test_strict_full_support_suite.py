from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.run_strict_full_v8_support_suite import reject_hidden_plan, statistics_command


def test_support_suite_rejects_hidden_plan_by_path(tmp_path) -> None:
    hidden_plan = tmp_path / "hidden_holdout_window_plan.json"
    hidden_plan.write_text(json.dumps({"selected_window_plan": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="hidden"):
        reject_hidden_plan(str(hidden_plan))


def test_support_suite_rejects_hidden_plan_by_split_name(tmp_path) -> None:
    plan = tmp_path / "window_plan.json"
    plan.write_text(
        json.dumps({"split_name": "hidden_holdout", "selected_window_plan": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hidden"):
        reject_hidden_plan(str(plan))


def test_statistics_command_allows_guard_attribution_candidate_label(tmp_path) -> None:
    command = statistics_command(
        rows_path=tmp_path / "benchmark_rows.csv",
        output_root=tmp_path / "stats",
        candidate_agent="all_guards",
        baseline_agents=["learned_core_only", "no_guard"],
        args=SimpleNamespace(bootstrap_samples=123),
    )

    candidate_index = command.index("--candidate_agent")
    baseline_index = command.index("--baseline_agents")
    assert command[candidate_index + 1] == "all_guards"
    assert command[baseline_index + 1 : baseline_index + 3] == ["learned_core_only", "no_guard"]
