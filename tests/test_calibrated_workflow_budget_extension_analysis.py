from __future__ import annotations

import pytest

from scripts.analyze_calibrated_workflow_budget_extension import _pair_rows


def test_pair_rows_matches_by_identity_when_inputs_are_reordered() -> None:
    short = [{"id": "a", "value": "short-a"}, {"id": "b", "value": "short-b"}]
    long = [{"id": "b", "value": "long-b"}, {"id": "a", "value": "long-a"}]
    pairs = _pair_rows(short, long, ("id",))
    assert [(left["id"], right["id"]) for left, right in pairs] == [("a", "a"), ("b", "b")]


def test_pair_rows_rejects_different_lengths_and_key_sets() -> None:
    with pytest.raises(RuntimeError, match="paired key-set mismatch"):
        _pair_rows([{"id": "a"}, {"id": "b"}], [{"id": "a"}], ("id",))


def test_pair_rows_rejects_missing_identity_field() -> None:
    with pytest.raises(RuntimeError, match="missing pairing key"):
        _pair_rows([{"id": "a"}], [{"value": "long-a"}], ("id",))


def test_pair_rows_rejects_duplicate_identity() -> None:
    with pytest.raises(RuntimeError, match="duplicate pairing key"):
        _pair_rows([{"id": "a"}, {"id": "a"}], [{"id": "a"}], ("id",))
