from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.evaluators.main_results_support import apply_frozen_window_plan


ROOT_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_strict_split_protocol",
    ROOT_DIR / "scripts" / "freeze_strict_split_protocol.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def window(offset: int, label: str) -> dict[str, object]:
    return {
        "window_id": f"window_{offset}",
        "frame_offset": offset,
        "window_length": 10,
        "window_class": label,
    }


def test_intervals_separated_enforces_extra_gap() -> None:
    assert MODULE.intervals_separated(window(0, "x"), window(15, "x"), 5)
    assert not MODULE.intervals_separated(window(0, "x"), window(14, "x"), 5)


def test_select_stratified_windows_balances_splits_and_audits() -> None:
    pools = {
        "active_non_mechanism_windows": [window(i, "active_non_mechanism") for i in range(0, 80, 20)],
        "mechanism_activating_windows": [window(i, "mechanism_activating") for i in range(100, 180, 20)],
        "idle_or_sparse_windows": [window(i, "idle_or_sparse") for i in range(200, 360, 20)],
    }
    plans = MODULE.select_stratified_windows(
        pools,
        split_names=("train", "dev"),
        targets_per_split={
            "active_non_mechanism": 1,
            "mechanism_activating": 1,
            "idle_or_sparse": 2,
        },
        minimum_gap_frames=5,
    )
    assert {name: len(items) for name, items in plans.items()} == {"train": 4, "dev": 4}
    assert MODULE.audit_split_plans(plans, minimum_gap_frames=5)["passed"] is True


def test_audit_rejects_cross_split_overlap() -> None:
    plans = {
        "formal": [window(0, "mechanism_activating")],
        "hidden_holdout": [window(8, "idle_or_sparse")],
    }
    audit = MODULE.audit_split_plans(plans, minimum_gap_frames=0)
    assert audit["passed"] is False
    assert audit["conflicts"][0]["left_split"] == "formal"


def test_apply_frozen_window_plan_records_provenance(tmp_path: Path) -> None:
    plan_path = tmp_path / "dev.json"
    plan_path.write_text(
        json.dumps(
            {
                "protocol_version": "test_v1",
                "split": "dev",
                "outcome_blind_selection": True,
                "selected_window_plan": [window(20, "mechanism_activating")],
            }
        ),
        encoding="utf-8",
    )
    payload = apply_frozen_window_plan({"selected_windows": []}, plan_path)
    assert payload["frozen_window_plan_protocol_version"] == "test_v1"
    assert payload["frozen_window_plan_split"] == "dev"
    assert payload["outcome_blind_selection"] is True
    assert payload["selected_windows"][0]["window_id"] == "window_20"
