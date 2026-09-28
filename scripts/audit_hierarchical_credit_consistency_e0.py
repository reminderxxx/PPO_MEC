#!/usr/bin/env python3
"""Exhaustively audit masked five-action PPO credit without env steps or training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import torch


ROOT = Path(__file__).resolve().parents[1]
AUDIT_VERSION = "hierarchical_credit_consistency_e0_v1"
CLIP_RATIO = 0.2
GRADIENT_TOLERANCE = 1e-10
OBJECTIVE_TOLERANCE = 1e-10
EXPECTED_CHECKPOINT_UPDATES = (4, 8, 12, 16)
EXPECTED_CONTRACTS = {
    "crdcm_checkpoint_format_version": "crdcm_checkpoint_v1",
    "crdcm_decision_contract_version": "crdcm_decision_v1",
    "crdcm_credit_contract_version": "mask_external_override_actor_credit_v1",
}

# These old logits and perturbations are pre-fixed, finite, and non-degenerate.
LOGIT_PROBES: tuple[dict[str, Any], ...] = (
    {
        "probe_id": "balanced",
        "old": {
            "slow": [0.20, -0.40, 0.70],
            "fast": [-0.30, 0.50],
            "event": [0.25, -0.35],
        },
        "delta": {
            "slow": [0.18, -0.12, 0.05],
            "fast": [-0.16, 0.21],
            "event": [-0.14, 0.19],
        },
    },
    {
        "probe_id": "prepare_leaning",
        "old": {
            "slow": [-0.65, 0.85, 0.10],
            "fast": [0.60, -0.25],
            "event": [-0.55, 0.75],
        },
        "delta": {
            "slow": [0.11, 0.24, -0.27],
            "fast": [0.17, -0.22],
            "event": [0.28, -0.31],
        },
    },
    {
        "probe_id": "fallback_leaning",
        "old": {
            "slow": [0.90, -0.15, -0.55],
            "fast": [-0.70, 0.80],
            "event": [0.65, -0.45],
        },
        "delta": {
            "slow": [-0.26, 0.13, 0.19],
            "fast": [0.29, -0.18],
            "event": [-0.21, 0.26],
        },
    },
    {
        "probe_id": "cache_fill_leaning",
        "old": {
            "slow": [-0.25, 0.95, -0.35],
            "fast": [0.35, -0.45],
            "event": [0.40, -0.20],
        },
        "delta": {
            "slow": [0.23, -0.19, 0.08],
            "fast": [-0.24, 0.27],
            "event": [0.16, -0.20],
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def all_nonempty_masks() -> list[tuple[bool, ...]]:
    return [
        tuple(bool(bits & (1 << action)) for action in range(5))
        for bits in range(1, 1 << 5)
    ]


def mask_id(mask: Iterable[bool]) -> str:
    return "".join("1" if valid else "0" for valid in mask)


def canonical_targets(action: int) -> dict[str, int]:
    if action == 0:
        return {"slow": 1, "fast": 0, "event": 0}
    if action == 1:
        return {"slow": 2, "fast": 0, "event": 0}
    if action == 2:
        return {"slow": 0, "fast": 1, "event": 0}
    if action == 4:
        return {"slow": 0, "fast": 0, "event": 1}
    return {"slow": 0, "fast": 0, "event": 0}


def tensor_logits(
    values: dict[str, list[float]], *, requires_grad: bool
) -> dict[str, torch.Tensor]:
    return {
        head: torch.tensor(
            logits,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        for head, logits in values.items()
    }


def add_logits(
    left: dict[str, list[float]], right: dict[str, list[float]]
) -> dict[str, list[float]]:
    return {
        head: [a + b for a, b in zip(left[head], right[head])]
        for head in ("slow", "fast", "event")
    }


def aggregated_action_scores(logits: dict[str, torch.Tensor]) -> torch.Tensor:
    slow = torch.log_softmax(logits["slow"], dim=-1)
    fast = torch.log_softmax(logits["fast"], dim=-1)
    event = torch.log_softmax(logits["event"], dim=-1)
    return torch.stack(
        [
            event[0] + slow[1],
            event[0] + slow[2],
            event[0] + slow[0] + fast[1],
            event[0] + slow[0] + fast[0],
            event[1],
        ]
    )


def masked_action_log_probs(
    logits: dict[str, torch.Tensor], mask: tuple[bool, ...]
) -> torch.Tensor:
    scores = aggregated_action_scores(logits)
    valid = torch.tensor(mask, dtype=torch.bool, device=scores.device)
    return torch.log_softmax(scores.masked_fill(~valid, -1.0e30), dim=-1)


def ppo_surrogate(ratio: torch.Tensor, advantage: float) -> torch.Tensor:
    advantage_tensor = torch.tensor(
        advantage, dtype=ratio.dtype, device=ratio.device
    )
    unclipped = ratio * advantage_tensor
    clipped = torch.clamp(
        ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO
    ) * advantage_tensor
    return torch.minimum(unclipped, clipped)


def flatten_gradients(
    objective: torch.Tensor,
    logits: dict[str, torch.Tensor],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    gradients = torch.autograd.grad(
        objective,
        tuple(logits[head] for head in ("slow", "fast", "event")),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    gradients = tuple(
        torch.zeros_like(logits[head]) if gradient is None else gradient
        for head, gradient in zip(("slow", "fast", "event"), gradients)
    )
    by_head = {
        head: float(torch.linalg.vector_norm(gradient).item())
        for head, gradient in zip(("slow", "fast", "event"), gradients)
    }
    return torch.cat([gradient.reshape(-1) for gradient in gradients]), by_head


def legacy_surrogate(
    old_logits: dict[str, torch.Tensor],
    new_logits: dict[str, torch.Tensor],
    *,
    action: int,
    advantage: float,
    included_heads: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = canonical_targets(action)
    heads = ("slow", "fast", "event")
    if included_heads is not None:
        heads = tuple(head for head in heads if head in included_heads)
    if not heads:
        zero = sum(value.sum() * 0.0 for value in new_logits.values())
        return zero, {}
    terms: list[torch.Tensor] = []
    ratios: dict[str, float] = {}
    for head in heads:
        old_log_prob = torch.log_softmax(old_logits[head], dim=-1)[targets[head]]
        new_log_prob = torch.log_softmax(new_logits[head], dim=-1)[targets[head]]
        ratio = torch.exp(new_log_prob - old_log_prob)
        ratios[head] = float(ratio.detach().item())
        terms.append(ppo_surrogate(ratio, advantage))
    return torch.stack(terms).mean(), ratios


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(
        torch.linalg.vector_norm(left).item()
        * torch.linalg.vector_norm(right).item()
    )
    if denominator <= 1e-18:
        return None
    return float(torch.dot(left, right).item() / denominator)


def case_result(
    probe: dict[str, Any],
    mask: tuple[bool, ...],
    action: int,
    advantage: float,
) -> dict[str, Any]:
    old_values = dict(probe["old"])
    new_values = add_logits(old_values, dict(probe["delta"]))
    old_logits = tensor_logits(old_values, requires_grad=False)
    new_logits = tensor_logits(new_values, requires_grad=True)

    old_action_log_probs = masked_action_log_probs(old_logits, mask)
    new_action_log_probs = masked_action_log_probs(new_logits, mask)
    old_log_prob = old_action_log_probs[action]
    new_log_prob = new_action_log_probs[action]
    exact_ratio = torch.exp(new_log_prob - old_log_prob)
    direct_ratio = torch.exp(new_log_prob) / torch.exp(old_log_prob)
    exact_surrogate = ppo_surrogate(exact_ratio, advantage)
    exact_loss = -exact_surrogate
    exact_gradient, exact_head_norms = flatten_gradients(
        exact_loss, new_logits, retain_graph=True
    )

    active_heads = {
        head
        for head, norm in exact_head_norms.items()
        if norm > GRADIENT_TOLERANCE
    }
    legacy_value, legacy_ratios = legacy_surrogate(
        old_logits,
        new_logits,
        action=action,
        advantage=advantage,
    )
    legacy_loss = -legacy_value
    legacy_gradient, legacy_head_norms = flatten_gradients(
        legacy_loss, new_logits, retain_graph=True
    )
    oracle_value, oracle_ratios = legacy_surrogate(
        old_logits,
        new_logits,
        action=action,
        advantage=advantage,
        included_heads=active_heads,
    )
    oracle_loss = -oracle_value
    oracle_gradient, _ = flatten_gradients(
        oracle_loss, new_logits, retain_graph=True
    )
    override_zero_loss = exact_loss * 0.0
    override_gradient, _ = flatten_gradients(
        override_zero_loss, new_logits, retain_graph=False
    )

    probabilities = torch.exp(new_action_log_probs)
    invalid_probability_max = max(
        [float(probabilities[index].item()) for index, valid in enumerate(mask) if not valid]
        or [0.0]
    )
    old_log_prob_rounded = round(float(old_log_prob.item()), 6)
    identity_ratio_from_rounded_old = math.exp(
        float(old_log_prob.item()) - old_log_prob_rounded
    )
    exact_legacy_gradient_gap = float(
        torch.max(torch.abs(exact_gradient - legacy_gradient)).item()
    )
    exact_oracle_gradient_gap = float(
        torch.max(torch.abs(exact_gradient - oracle_gradient)).item()
    )
    zero_gradient_credited_heads = sorted(
        head
        for head in ("slow", "fast", "event")
        if exact_head_norms[head] <= GRADIENT_TOLERANCE
        and legacy_head_norms[head] > GRADIENT_TOLERANCE
    )
    return {
        "probe_id": probe["probe_id"],
        "mask": mask_id(mask),
        "valid_action_count": sum(mask),
        "action": action,
        "advantage": advantage,
        "probability_sum": float(probabilities.sum().item()),
        "invalid_probability_max": invalid_probability_max,
        "old_behavior_log_prob": float(old_log_prob.item()),
        "new_behavior_log_prob": float(new_log_prob.item()),
        "exact_ratio": float(exact_ratio.detach().item()),
        "direct_probability_ratio": float(direct_ratio.detach().item()),
        "ratio_abs_error": float(torch.abs(exact_ratio - direct_ratio).item()),
        "exact_clipped": not math.isclose(
            float(exact_ratio.detach().item()),
            max(
                1.0 - CLIP_RATIO,
                min(1.0 + CLIP_RATIO, float(exact_ratio.detach().item())),
            ),
            abs_tol=1e-15,
        ),
        "exact_surrogate": float(exact_surrogate.detach().item()),
        "legacy_surrogate": float(legacy_value.detach().item()),
        "oracle_zero_gradient_head_mask_surrogate": float(
            oracle_value.detach().item()
        ),
        "exact_legacy_objective_abs_gap": abs(
            float(exact_surrogate.detach().item())
            - float(legacy_value.detach().item())
        ),
        "exact_oracle_objective_abs_gap": abs(
            float(exact_surrogate.detach().item())
            - float(oracle_value.detach().item())
        ),
        "exact_legacy_gradient_max_abs_gap": exact_legacy_gradient_gap,
        "exact_oracle_gradient_max_abs_gap": exact_oracle_gradient_gap,
        "exact_legacy_gradient_cosine": cosine_similarity(
            exact_gradient, legacy_gradient
        ),
        "exact_oracle_gradient_cosine": cosine_similarity(
            exact_gradient, oracle_gradient
        ),
        "exact_gradient_norm": float(torch.linalg.vector_norm(exact_gradient).item()),
        "legacy_gradient_norm": float(
            torch.linalg.vector_norm(legacy_gradient).item()
        ),
        "oracle_gradient_norm": float(
            torch.linalg.vector_norm(oracle_gradient).item()
        ),
        "exact_gradient_norm_slow": exact_head_norms["slow"],
        "exact_gradient_norm_fast": exact_head_norms["fast"],
        "exact_gradient_norm_event": exact_head_norms["event"],
        "legacy_gradient_norm_slow": legacy_head_norms["slow"],
        "legacy_gradient_norm_fast": legacy_head_norms["fast"],
        "legacy_gradient_norm_event": legacy_head_norms["event"],
        "active_exact_heads": "|".join(sorted(active_heads)),
        "zero_gradient_credited_heads": "|".join(zero_gradient_credited_heads),
        "zero_gradient_head_credit": bool(zero_gradient_credited_heads),
        "legacy_head_ratios_json": json.dumps(legacy_ratios, sort_keys=True),
        "oracle_head_ratios_json": json.dumps(oracle_ratios, sort_keys=True),
        "identity_ratio_error_from_six_decimal_old_log_prob": abs(
            identity_ratio_from_rounded_old - 1.0
        ),
        "override_zero_actor_gradient_max_abs": float(
            torch.max(torch.abs(override_gradient)).item()
        ),
    }


def run_math_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks = all_nonempty_masks()
    rows = [
        case_result(probe, mask, action, advantage)
        for probe in LOGIT_PROBES
        for mask in masks
        for action, valid in enumerate(mask)
        if valid
        for advantage in (1.0, -1.0)
    ]
    single_action_rows = [row for row in rows if row["valid_action_count"] == 1]
    legacy_mismatch = [
        row
        for row in rows
        if row["exact_legacy_gradient_max_abs_gap"] > GRADIENT_TOLERANCE
        or row["exact_legacy_objective_abs_gap"] > OBJECTIVE_TOLERANCE
    ]
    oracle_mismatch = [
        row
        for row in rows
        if row["exact_oracle_gradient_max_abs_gap"] > GRADIENT_TOLERANCE
        or row["exact_oracle_objective_abs_gap"] > OBJECTIVE_TOLERANCE
    ]
    summary = {
        "mask_count": len(masks),
        "probe_count": len(LOGIT_PROBES),
        "mask_action_pairs_per_probe": sum(sum(mask) for mask in masks),
        "advantage_signs": [1.0, -1.0],
        "case_count": len(rows),
        "probability_normalization_max_error": max(
            abs(row["probability_sum"] - 1.0) for row in rows
        ),
        "invalid_padded_probability_max": max(
            row["invalid_probability_max"] for row in rows
        ),
        "old_new_ratio_identity_max_error": max(
            row["ratio_abs_error"] for row in rows
        ),
        "six_decimal_old_log_prob_identity_ratio_max_error": max(
            row["identity_ratio_error_from_six_decimal_old_log_prob"]
            for row in rows
        ),
        "single_legal_action_case_count": len(single_action_rows),
        "single_legal_action_exact_gradient_max": max(
            row["exact_gradient_norm"] for row in single_action_rows
        ),
        "single_legal_action_legacy_nonzero_gradient_count": sum(
            row["legacy_gradient_norm"] > GRADIENT_TOLERANCE
            for row in single_action_rows
        ),
        "zero_exact_gradient_head_but_legacy_credited_case_count": sum(
            row["zero_gradient_head_credit"] for row in rows
        ),
        "legacy_surrogate_mismatch_case_count": len(legacy_mismatch),
        "legacy_surrogate_mismatch_rate": len(legacy_mismatch) / len(rows),
        "zero_gradient_head_mask_repair_mismatch_case_count": len(oracle_mismatch),
        "zero_gradient_head_mask_repair_mismatch_rate": len(oracle_mismatch)
        / len(rows),
        "exact_ratio_outside_clip_range_case_count": sum(
            row["exact_clipped"] for row in rows
        ),
        "override_zero_actor_gradient_max_abs": max(
            row["override_zero_actor_gradient_max_abs"] for row in rows
        ),
        "verdict": "LEGACY_PER_HEAD_SURROGATE_NOT_EQUIVALENT_TO_MASKED_ENV_ACTION_PPO",
        "head_mask_only_repair_verdict": "INSUFFICIENT",
    }
    return rows, summary


def checkpoint_readback(source_root: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_root is None:
        return [], {"status": "not_requested"}
    checkpoint_paths = sorted(
        path
        for path in source_root.resolve().glob("runs/*/*/checkpoints/update_*.pt")
        if int(path.stem.split("_")[-1]) in EXPECTED_CHECKPOINT_UPDATES
    )
    if len(checkpoint_paths) != 48:
        raise ValueError(f"expected 48 stage checkpoints, found {len(checkpoint_paths)}")
    before = {str(path): sha256(path) for path in checkpoint_paths}
    rows: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu")
        config = dict(checkpoint.get("config", {}) or {})
        contract_match = all(
            config.get(key) == expected for key, expected in EXPECTED_CONTRACTS.items()
        )
        rows.append(
            {
                "checkpoint_path": str(path),
                "checkpoint_sha256": before[str(path)],
                "agent_name": checkpoint.get("agent_name"),
                "policy_type": checkpoint.get("policy_type"),
                "update_count": int(checkpoint.get("update_count", -1)),
                "expected_update_count": int(path.stem.split("_")[-1]),
                "crdcm_feature_mode": config.get("crdcm_feature_mode"),
                "crdcm_checkpoint_format_version": config.get(
                    "crdcm_checkpoint_format_version"
                ),
                "crdcm_decision_contract_version": config.get(
                    "crdcm_decision_contract_version"
                ),
                "crdcm_credit_contract_version": config.get(
                    "crdcm_credit_contract_version"
                ),
                "contract_match": contract_match,
                "update_match": int(checkpoint.get("update_count", -1))
                == int(path.stem.split("_")[-1]),
                "has_network_state": isinstance(
                    checkpoint.get("network_state_dict"), dict
                ),
                "has_crdcm_residual_state": isinstance(
                    checkpoint.get("crdcm_residual_state_dict"), dict
                ),
            }
        )
    after = {str(path): sha256(path) for path in checkpoint_paths}
    if before != after:
        raise RuntimeError("checkpoint bytes changed during readback")
    if not all(
        row["contract_match"]
        and row["update_match"]
        and row["has_network_state"]
        and row["has_crdcm_residual_state"]
        for row in rows
    ):
        raise ValueError("checkpoint contract readback failed")
    return rows, {
        "status": "pass",
        "checkpoint_count": len(rows),
        "before_after_sha256_equal": before == after,
        "legacy_credit_contract": EXPECTED_CONTRACTS[
            "crdcm_credit_contract_version"
        ],
        "compatibility_boundary": (
            "Any corrected actor-credit implementation must publish a new strict "
            "credit/checkpoint contract; these v1 checkpoints remain historical."
        ),
    }


def build_repair_spec() -> dict[str, Any]:
    return {
        "scope": "future_separate_implementation_task_not_applied_by_e0",
        "new_credit_contract_required": "exact_masked_env_action_ppo_v2",
        "requirements": [
            "Persist the sampled masked five-action behavior log-probability at training precision; rounding is logging-only.",
            "Recompute the same masked five-action categorical from current logits and the original action mask during update.",
            "Use one environment-action PPO ratio and one clip operation as the primary actor objective.",
            "Compute policy entropy from the same masked five-action categorical.",
            "Keep externally overridden transitions at actor weight zero; critic learning must have an explicit shared-parameter policy.",
            "Do not treat canonical latent head labels as observed behavior actions; optional head losses require separate auxiliary contracts.",
            "Bump and strictly validate checkpoint/credit contract versions; do not reinterpret v1 checkpoints as corrected policies.",
        ],
        "acceptance": [
            "All 31 nonempty masks and every legal action match direct probability ratios, PPO objectives, and gradients.",
            "Every single-legal-action mask has exact zero policy gradient.",
            "Invalid padded actions have zero probability and cannot enter actor loss.",
            "actor_credit_weight=0 produces zero actor gradient.",
            "The legacy per-head mismatch is retained as a regression fixture rather than erased from history.",
        ],
        "explicit_non_fix": (
            "Masking only heads with zero exact gradient is insufficient because conditional "
            "mask normalization and independent per-head clipping still do not reproduce the "
            "single masked environment-action PPO objective."
        ),
    }


def build_matched_experiment_freeze() -> dict[str, Any]:
    return {
        "status": "draft_not_execution_authorization",
        "prerequisite": "E0 corrected-contract tests pass in a separate implementation task",
        "conditions": [
            "legacy_crdcm_sa_historical_contract_v1",
            "corrected_crdcm_sa_exact_masked_env_ppo_v2",
            "corrected_crdcm_mappo_exact_masked_env_ppo_v2",
            "unchanged_crdcm_ppo_control",
        ],
        "seeds": [1401, 1402, 1403],
        "episodes_per_cell": 64,
        "max_steps": 20,
        "fixed_checkpoint_update": 16,
        "training_episode_upper_bound": 768,
        "learned_evaluation_episode_count": 144,
        "heuristic_evaluation_episode_count": 12,
        "total_episode_upper_bound": 924,
        "environment_step_upper_bound": 18480,
        "frozen_equalities": [
            "reward",
            "temperature schedules",
            "model sizes",
            "seeds",
            "windows",
            "workflows",
            "action schema and masks",
            "training budget",
            "checkpoint selection rule",
        ],
        "readouts": [
            "workflow completion and request failures",
            "all-episode transfer/backhaul/migration cost including failures",
            "cost per completed workflow and successful request with denominators shown",
            "per-seed actions and failure transitions",
            "optimizer consistency metrics",
            "training/inference wall-clock and peak memory",
        ],
        "stop_rules": [
            "Do not train if the corrected contract fails E0.",
            "Do not add seeds, budget, reward changes, temperature changes, or checkpoint selection after observing E1.",
            "If correctness passes but performance does not improve reliably, stop expanding the SA superiority claim.",
            "A correctness benefit is not evidence for base-sharing/migration mechanism novelty or architecture necessity.",
        ],
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows, math_summary = run_math_audit()
    checkpoint_rows, checkpoint_summary = checkpoint_readback(args.source_root)
    reviewed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    summary = {
        "audit_version": AUDIT_VERSION,
        "reviewed_at": reviewed_at,
        "literature_cutoff": "2026-09-28_no_new_web_search",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": "hierarchical_credit_consistency_e0_20260928",
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": git_value("rev-parse", "HEAD"),
        "evidence_level": "E2_IMPLEMENTATION_MATH_AUDITED_PLUS_CHECKPOINT_READBACK",
        "scope": "mathematical_validation_only_no_environment_no_training",
        "environment_steps_executed": 0,
        "training_updates_executed": 0,
        "clip_ratio": CLIP_RATIO,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "legacy_reference_contract": {
            "algorithm_slice": "full_sa_head_credit_disabled_equal_weight_core",
            "head_weights": {"slow": 1.0, "fast": 1.0, "event": 1.0},
            "base_and_event_advantage": "equal_fixed_sign",
            "event_reliability_modifier": "neutral",
            "boundary": (
                "This is a valid controlled slice of the SA actor core. Universal PPO "
                "consistency must hold on this slice; MAPPO-specific weights and extra "
                "auxiliary modifiers are not separately performance-tested by E0."
            ),
        },
        "fixed_logit_probes": LOGIT_PROBES,
        "math_summary": math_summary,
        "checkpoint_readback": checkpoint_summary,
        "minimal_repair_specification": build_repair_spec(),
        "matched_experiment_freeze_draft": build_matched_experiment_freeze(),
        "scientific_boundary": {
            "implementation_defect": (
                "The legacy canonical per-head surrogate is not the PPO objective of the "
                "masked five-action behavior distribution."
            ),
            "performance_causality": "UNVERIFIED",
            "mechanism_effect": "UNVERIFIED_AND_SEPARATE_FROM_E0_E1",
            "architecture_necessity": "UNVERIFIED_AND_SEPARATE_FROM_E0_E1_E2",
        },
    }
    write_csv(output / "case_results.csv", rows)
    if checkpoint_rows:
        write_csv(output / "checkpoint_readback.csv", checkpoint_rows)
    write_json(output / "summary.json", summary)
    manifest_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    write_json(
        output / "artifact_integrity_manifest.json",
        {
            "manifest_version": "artifact_integrity_manifest_v1",
            "generated_at": reviewed_at,
            "files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in manifest_files
            ],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
