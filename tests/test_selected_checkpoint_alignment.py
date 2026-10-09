from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import analyze_calibrated_workflow_selected_checkpoint_alignment as alignment


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | {
            key for child in value.values() for key in _nested_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _nested_keys(child)}
    return set()


def test_state_freeze_is_outcome_blind_public_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "frozen_states"
    alignment.freeze_states(output)
    manifest = json.loads((output / "state_manifest.json").read_text(encoding="utf-8"))

    records = manifest["records"]
    assert manifest["checkpoint_or_evaluation_inputs_used"] is False
    assert manifest["future_actual_link_information_used"] is False
    assert len(records) == 48
    assert len({row["state_sha256"] for row in records}) == 48
    assert {
        design_id: sum(row["design_id"] == design_id for row in records)
        for design_id in {row["design_id"] for row in records}
    } == {"dev_00": 12, "dev_01": 12, "dev_02": 12, "dev_03": 12}
    assert all(row["probe"]["complete_continuation"] for row in records)
    assert all(
        row["probe"]["target_semantics"]
        == "complete_fixed_behavior_discounted_return_not_vpi"
        for row in records
    )
    assert not any(
        alignment.FORBIDDEN_PUBLIC_KEYS & _nested_keys(row["public_state"])
        for row in records
    )


def test_read_only_analyzer_has_no_learning_call_and_hashes_are_stable() -> None:
    source = Path(inspect.getsourcefile(alignment) or "").read_text(encoding="utf-8")
    assert ".learn(" not in source

    payload = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "state": {2: [torch.tensor(5.0), "fixed"]},
    }
    first = alignment._state_dict_hash(payload)
    second = alignment._state_dict_hash(payload)
    assert first == second
    payload["weight"][0, 0] = 9.0
    assert alignment._state_dict_hash(payload) != first
