from __future__ import annotations

import torch

from scripts.audit_compute_cost import summarize, tensor_count


def test_tensor_count_recurses_through_model_ensemble_state() -> None:
    payload = {
        "models": [
            {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)},
            {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)},
        ]
    }

    assert tensor_count(payload) == 16


def test_summarize_marks_absent_measurement_unavailable() -> None:
    assert summarize([]) == {"status": "unavailable", "sample_count": 0}
    measured = summarize([1.0, 2.0, 3.0])
    assert measured["status"] == "measured"
    assert measured["mean"] == 2.0
    assert measured["p95"] == 3.0
