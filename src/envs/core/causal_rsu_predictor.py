"""Small train-only RSU transition predictor for the development interface.

Inference accepts a visible prefix only. The fitted transition table is
deliberately low capacity; unseen states return an explicit unknown forecast.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _key(prefix: list[str]) -> str:
    current = prefix[-1]
    streak = 1
    for item in reversed(prefix[:-1]):
        if item != current:
            break
        streak += 1
    return f"{current}|{min(streak, 4)}"


def fit_predictor(sequences: list[list[str]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            prefix = [str(item) for item in sequence[: index + 1]]
            counts[_key(prefix)][str(sequence[index + 1])] += 1
    table = {key: dict(sorted(counter.items())) for key, counter in sorted(counts.items())}
    body = {"schema_version": "prefix_transition_counts_v1", "min_support": 2, "counts": table}
    return {**body, "sha256": canonical_hash(body)}


def forecast(prefix: list[str], model: dict[str, Any], horizon: int) -> dict[str, Any]:
    if not prefix or horizon < 1:
        raise ValueError("forecast requires a nonempty prefix and positive horizon")
    if model.get("sha256") != canonical_hash({key: value for key, value in model.items() if key != "sha256"}):
        raise RuntimeError("causal predictor hash mismatch")
    visible = [str(item) for item in prefix]
    result: list[str] = []
    supports: list[int] = []
    confidences: list[float] = []
    for _ in range(horizon):
        counts = model["counts"].get(_key(visible), {})
        total = sum(int(value) for value in counts.values())
        if total < int(model["min_support"]):
            break
        choice = min(counts, key=lambda item: (-int(counts[item]), item))
        result.append(choice)
        supports.append(total)
        confidences.append(int(counts[choice]) / total)
        visible.append(choice)
    return {
        "sequence": result,
        "status": "known" if result else "unknown",
        "confidence": min(confidences) if confidences else 0.0,
        "support": min(supports) if supports else 0,
        "prefix_end_index": len(prefix) - 1,
        "prefix_sha256": canonical_hash([str(item) for item in prefix]),
        "predictor_sha256": model["sha256"],
    }
