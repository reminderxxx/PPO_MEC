"""Frozen mobility plans must bind the actual loaded source before rollout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.evaluators import main_results_support


PLAN = {
    "window_id": "window_peachtree_off244_len24_t1163059400_1163061700",
    "source_segment_id": "peachtree",
    "source_location": "peachtree",
    "frame_offset": 244,
    "window_length": 24,
    "segment_frame_start": 244,
    "segment_frame_end": 267,
    "time_index_start": 1163059400,
    "time_index_end": 1163061700,
}


def _load(metadata: dict, *, expected: dict | None = PLAN):
    with patch.object(
        main_results_support,
        "load_real_mobility_bundle",
        return_value=SimpleNamespace(rsu_metadata=metadata),
    ):
        return main_results_support.load_window_bundle(
            root_dir=Path("."),
            mobility_csv_path="unused.csv",
            max_mobility_rows=2500,
            rsu_layout="auto_dominant_tight",
            frame_offset=244,
            window_length=24,
            random_seed=7,
            expected_window_id=PLAN["window_id"] if expected else "",
            expected_window_identity=expected,
        )


def test_exact_frozen_window_identity_passes() -> None:
    assert _load(dict(PLAN)).rsu_metadata == PLAN


def test_same_offset_wrong_source_is_rejected_before_rollout() -> None:
    observed = dict(PLAN)
    observed.update(
        window_id="window_lankershim_off244_len24_t1118935704600_1118935706900",
        source_segment_id="lankershim",
        source_location="lankershim",
        time_index_start=1118935704600,
        time_index_end=1118935706900,
    )
    with pytest.raises(ValueError, match="window_id"):
        _load(observed)


@pytest.mark.parametrize(
    "field",
    ["source_segment_id", "source_location", "segment_frame_end", "time_index_start", "window_length"],
)
def test_frozen_identity_mismatch_is_rejected(field: str) -> None:
    observed = dict(PLAN)
    observed[field] = observed[field] + 1 if isinstance(observed[field], int) else "lankershim"
    with pytest.raises(ValueError, match=field):
        _load(observed)


def test_unfrozen_loader_keeps_existing_behavior() -> None:
    assert _load({"window_id": "unfrozen"}, expected=None).rsu_metadata["window_id"] == "unfrozen"
