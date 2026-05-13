"""指标聚合工具。"""

from __future__ import annotations

from typing import Any


def 求和(values: list[float | int]) -> float:
    """返回数值列表求和。"""
    return float(sum(values)) if values else 0.0


def 安全均值(values: list[float | int]) -> float:
    """返回数值列表均值。"""
    return 求和(values) / float(len(values)) if values else 0.0


def 安全比率(numerator: float | int, denominator: float | int) -> float:
    """返回安全比率。"""
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def 统计布尔数量(values: list[bool]) -> dict[str, int]:
    """统计布尔列表中的真假数量。"""
    true_count = sum(1 for item in values if item)
    false_count = sum(1 for item in values if not item)
    return {
        "true": int(true_count),
        "false": int(false_count),
        "total": int(len(values)),
    }


def 聚合奖励拆解(reward_dicts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """聚合 reward breakdown。"""
    if not reward_dicts:
        return {}
    field_names = sorted(reward_dicts[0].keys())
    aggregated: dict[str, dict[str, float]] = {}
    for field_name in field_names:
        values = [float(item.get(field_name, 0.0)) for item in reward_dicts]
        aggregated[field_name] = {
            "sum": round(求和(values), 6),
            "mean": round(安全均值(values), 6),
            "min": round(min(values), 6) if values else 0.0,
            "max": round(max(values), 6) if values else 0.0,
        }
    return aggregated
