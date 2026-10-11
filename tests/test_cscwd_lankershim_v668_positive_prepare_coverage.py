"""Pure contracts for the frozen v668 positive-prepare coverage probe."""

import json
from pathlib import Path

from scripts.probe_cscwd_lankershim_v668_positive_prepare_coverage import (
    _gate,
    _load_protocol,
)


def test_v668_protocol_is_bounded_and_reads_only_train_role() -> None:
    path = Path(
        "configs/experiment/cscwd_lankershim_v668_positive_prepare_coverage_postfix_v1.json"
    ).resolve()
    protocol, split, _ = _load_protocol(path)
    assert protocol["environment_step_budget"] == 192
    assert protocol["selection_and_check_rows_read"] is False
    assert protocol["optimizer_steps"] == 0
    assert [row["role"] for row in split["selected_windows"]] == [
        "train", "selection", "check"
    ]


def test_gate_requires_real_intervention_and_consumer_identity() -> None:
    protocol = {
        "minimum_base_positive_prepare_count": 1,
        "minimum_candidate_prepare_to_abstain_count": 1,
    }
    row = {
        "base_decision": "prepare", "base_supervision_weight": 1.0,
        "candidate_decision": "abstain", "candidate_supervision_weight": 0.0,
        "base_actual_consumer_match": True,
        "candidate_actual_consumer_match": True,
    }
    assert _gate([row], protocol)["coverage_gate_passed"] is True
    mismatch = json.loads(json.dumps(row))
    mismatch["candidate_actual_consumer_match"] = False
    assert _gate([mismatch], protocol)["coverage_gate_passed"] is False
