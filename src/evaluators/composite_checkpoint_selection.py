from __future__ import annotations

from typing import Any


def normalize_reward(reward: float, reward_min: float, reward_max: float) -> float:
    reward = float(reward)
    reward_min = float(reward_min)
    reward_max = float(reward_max)
    if reward_max - reward_min <= 1e-9:
        return 0.0
    return max(0.0, min((reward - reward_min) / (reward_max - reward_min), 1.0))


def compute_composite_heuristic_score(
    metrics: dict[str, Any],
    *,
    reward_min: float,
    reward_max: float,
    alpha: float = 2.0,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> dict[str, float]:
    reward = float(metrics.get("total_reward", 0.0))
    continuity = float(metrics.get("workflow_continuity_rate", 0.0))
    mechanism_realization_rate = float(metrics.get("mechanism_realization_rate", 0.0))
    handoff_ready_ratio = float(metrics.get("handoff_ready_ratio", 0.0))
    normalized_reward = normalize_reward(
        reward=reward,
        reward_min=reward_min,
        reward_max=reward_max,
    )
    composite_score = (
        normalized_reward
        + float(alpha) * mechanism_realization_rate
        + float(beta) * handoff_ready_ratio
        + float(gamma) * continuity
    )
    return {
        "score": round(composite_score, 6),
        "normalized_reward": round(normalized_reward, 6),
        "reward": round(reward, 6),
        "workflow_continuity_rate": round(continuity, 6),
        "mechanism_realization_rate": round(mechanism_realization_rate, 6),
        "handoff_ready_ratio": round(handoff_ready_ratio, 6),
        "alpha": round(float(alpha), 6),
        "beta": round(float(beta), 6),
        "gamma": round(float(gamma), 6),
    }


def annotate_candidates_with_composite_score(
    candidates: list[dict[str, Any]],
    *,
    metrics_key: str = "selection_metrics",
    alpha: float = 2.0,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> list[dict[str, Any]]:
    reward_values = [
        float(candidate.get(metrics_key, {}).get("total_reward", 0.0))
        for candidate in candidates
    ]
    reward_min = min(reward_values) if reward_values else 0.0
    reward_max = max(reward_values) if reward_values else 0.0
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_copy = dict(candidate)
        candidate_copy["composite_score"] = compute_composite_heuristic_score(
            dict(candidate.get(metrics_key, {})),
            reward_min=reward_min,
            reward_max=reward_max,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        annotated.append(candidate_copy)
    return annotated


def select_best_composite_candidate(
    candidates: list[dict[str, Any]],
    *,
    metrics_key: str = "selection_metrics",
    alpha: float = 2.0,
    beta: float = 2.0,
    gamma: float = 1.0,
    min_mechanism_realization_rate: float = 0.3,
    min_handoff_ready_ratio: float = 1e-9,
) -> dict[str, Any]:
    annotated = annotate_candidates_with_composite_score(
        candidates,
        metrics_key=metrics_key,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    eligible = [
        candidate
        for candidate in annotated
        if float(candidate.get(metrics_key, {}).get("mechanism_realization_rate", 0.0)) > float(min_mechanism_realization_rate)
        and float(candidate.get(metrics_key, {}).get("handoff_ready_ratio", 0.0)) > float(min_handoff_ready_ratio)
    ]
    selection_pool = eligible or annotated
    if not selection_pool:
        raise RuntimeError("No checkpoint candidates available for composite selection.")
    best_candidate = max(
        selection_pool,
        key=lambda candidate: (
            float(candidate.get("composite_score", {}).get("score", float("-inf"))),
            float(candidate.get(metrics_key, {}).get("mechanism_realization_rate", 0.0)),
            float(candidate.get(metrics_key, {}).get("handoff_ready_ratio", 0.0)),
            float(candidate.get(metrics_key, {}).get("workflow_continuity_rate", 0.0)),
            float(candidate.get(metrics_key, {}).get("total_reward", 0.0)),
            int(candidate.get("update_index", 0) or 0),
        ),
    )
    return {
        "annotated_candidates": annotated,
        "eligible_candidates": eligible,
        "selected_from_eligible_pool": bool(eligible),
        "best_candidate": best_candidate,
        "selection_formula": "Norm(Reward) + alpha * mechanism_realization_rate + beta * handoff_ready_ratio + gamma * continuity",
        "selection_weights": {
            "alpha": round(float(alpha), 6),
            "beta": round(float(beta), 6),
            "gamma": round(float(gamma), 6),
        },
        "eligibility_thresholds": {
            "min_mechanism_realization_rate": round(float(min_mechanism_realization_rate), 6),
            "min_handoff_ready_ratio": round(float(min_handoff_ready_ratio), 6),
        },
    }
