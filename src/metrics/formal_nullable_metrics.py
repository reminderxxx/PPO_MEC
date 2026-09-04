"""Strict nullable-metric semantics shared by the active formal command graph."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


FORMAL_NULLABLE_METRIC_AGGREGATION_CONTRACT_VERSION = "1.0.0"
END_TO_END_WORKFLOW_DELAY = "end_to_end_workflow_delay"
AVAILABLE_COMPLETED_WORKFLOW = "available_completed_workflow"
UNAVAILABLE_DELAY_REASONS = frozenset(
    {
        "unavailable_failed_or_incomplete_workflow",
        "unavailable_right_censored_workflow",
    }
)


class NullableMetricContractError(ValueError):
    """Raised when a formal nullable value violates the frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NullableMetricContractError("value is not strict finite JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def nullable_finite_value(
    value: Any,
    *,
    field: str,
    csv_empty_is_null: bool = False,
    numeric_strings_allowed: bool = False,
) -> float | None:
    """Return a finite scalar or ``None`` without ever substituting a sentinel."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise NullableMetricContractError(f"bool is invalid for nullable metric: {field}")
    if isinstance(value, str):
        if value == "" and csv_empty_is_null:
            return None
        if not numeric_strings_allowed:
            raise NullableMetricContractError(
                f"string is invalid for nullable metric: {field}"
            )
        try:
            value = float(value)
        except ValueError as exc:
            raise NullableMetricContractError(
                f"unparseable string for nullable metric: {field}"
            ) from exc
    if not isinstance(value, (int, float)):
        raise NullableMetricContractError(
            f"unsupported value type for nullable metric: {field}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise NullableMetricContractError(f"non-finite nullable metric: {field}")
    return result


def reduce_nullable_values(
    values: Iterable[Any],
    *,
    field: str,
    csv_empty_is_null: bool = False,
    numeric_strings_allowed: bool = False,
    precision: int = 6,
) -> dict[str, Any]:
    materialized = list(values)
    available = [
        parsed
        for value in materialized
        if (
            parsed := nullable_finite_value(
                value,
                field=field,
                csv_empty_is_null=csv_empty_is_null,
                numeric_strings_allowed=numeric_strings_allowed,
            )
        )
        is not None
    ]
    total_count = len(materialized)
    available_count = len(available)
    unavailable_count = total_count - available_count
    if total_count == 0:
        status = "unavailable_no_rows"
        mean = None
    elif available_count == 0:
        status = "unavailable_all_values"
        mean = None
    elif unavailable_count:
        status = "available_partial"
        mean = round(fmean(available), precision)
    else:
        status = "available_complete"
        mean = round(fmean(available), precision)
    return {
        "total_count": total_count,
        "available_count": available_count,
        "unavailable_count": unavailable_count,
        "availability_status": status,
        "mean": mean,
    }


def reduce_nullable_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    required_fields: bool = True,
    csv_empty_is_null: bool = False,
    numeric_strings_allowed: bool = False,
    precision: int = 6,
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    means: dict[str, float | None] = {}
    availability: dict[str, dict[str, Any]] = {}
    for field in metric_names:
        values: list[Any] = []
        for index, row in enumerate(rows):
            if field not in row:
                if required_fields:
                    raise NullableMetricContractError(
                        f"required nullable metric missing: {field}; row={index}"
                    )
                values.append(None)
            else:
                values.append(row[field])
        stats = reduce_nullable_values(
            values,
            field=field,
            csv_empty_is_null=csv_empty_is_null,
            numeric_strings_allowed=numeric_strings_allowed,
            precision=precision,
        )
        means[field] = stats.pop("mean")
        availability[field] = stats
    canonical_json_bytes({"mean_metrics": means, "metric_availability": availability})
    return means, availability


def validate_end_to_end_delay_reason(
    value: Any,
    reason: Any,
    *,
    field: str = END_TO_END_WORKFLOW_DELAY,
) -> None:
    parsed = nullable_finite_value(value, field=field)
    if parsed is not None and reason != AVAILABLE_COMPLETED_WORKFLOW:
        raise NullableMetricContractError(
            "finite end-to-end delay requires available_completed_workflow"
        )
    if parsed is None and reason not in UNAVAILABLE_DELAY_REASONS:
        raise NullableMetricContractError(
            "null end-to-end delay requires failed/incomplete/right-censored reason"
        )


__all__ = [
    "AVAILABLE_COMPLETED_WORKFLOW",
    "END_TO_END_WORKFLOW_DELAY",
    "FORMAL_NULLABLE_METRIC_AGGREGATION_CONTRACT_VERSION",
    "NullableMetricContractError",
    "UNAVAILABLE_DELAY_REASONS",
    "canonical_json_bytes",
    "canonical_sha256",
    "nullable_finite_value",
    "reduce_nullable_metric_rows",
    "reduce_nullable_values",
    "validate_end_to_end_delay_reason",
]
