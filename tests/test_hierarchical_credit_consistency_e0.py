from __future__ import annotations

import math

import torch

from scripts.audit_hierarchical_credit_consistency_e0 import (
    GRADIENT_TOLERANCE,
    LOGIT_PROBES,
    add_logits,
    all_nonempty_masks,
    case_result,
    masked_action_log_probs,
    run_math_audit,
    tensor_logits,
)


def test_all_31_masks_are_normalized_and_padding_has_zero_mass() -> None:
    masks = all_nonempty_masks()
    assert len(masks) == 31
    for probe in LOGIT_PROBES:
        values = add_logits(probe["old"], probe["delta"])
        logits = tensor_logits(values, requires_grad=False)
        for mask in masks:
            probabilities = torch.exp(masked_action_log_probs(logits, mask))
            assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12)
            for action, valid in enumerate(mask):
                if not valid:
                    assert float(probabilities[action]) == 0.0


def test_single_legal_action_has_zero_exact_gradient_but_legacy_does_not() -> None:
    legacy_nonzero = 0
    for probe in LOGIT_PROBES:
        for action in range(5):
            mask = tuple(index == action for index in range(5))
            for advantage in (1.0, -1.0):
                row = case_result(probe, mask, action, advantage)
                assert row["exact_ratio"] == 1.0
                assert row["exact_gradient_norm"] <= GRADIENT_TOLERANCE
                assert row["override_zero_actor_gradient_max_abs"] == 0.0
                legacy_nonzero += row["legacy_gradient_norm"] > GRADIENT_TOLERANCE
    assert legacy_nonzero > 0


def test_legacy_surrogate_and_head_mask_only_repair_are_not_exact() -> None:
    rows, summary = run_math_audit()
    assert len(rows) == 640
    assert summary["mask_count"] == 31
    assert summary["mask_action_pairs_per_probe"] == 80
    assert summary["legacy_surrogate_mismatch_case_count"] > 0
    assert summary["zero_gradient_head_mask_repair_mismatch_case_count"] > 0
    assert summary["head_mask_only_repair_verdict"] == "INSUFFICIENT"


def test_ratio_old_logprob_and_override_boundaries() -> None:
    rows, summary = run_math_audit()
    assert summary["old_new_ratio_identity_max_error"] <= 1e-12
    assert summary["six_decimal_old_log_prob_identity_ratio_max_error"] < 1e-6
    assert summary["override_zero_actor_gradient_max_abs"] == 0.0
    assert any(row["exact_clipped"] for row in rows)
    assert any(row["zero_gradient_head_credit"] for row in rows)
